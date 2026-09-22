# -*- coding: utf-8 -*-
"""PVTv2-B4 + WS-SAM-style decoder for RCLCOD."""
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone.pvt_v2_eff import pvt_v2_eff_b4
from .zoomnext.ops import PixelNormalizer

LOGGER = logging.getLogger("main")

class BasicConv(nn.Module):
    def __init__(self, cin, cout, k, stride=1, padding=0, dilation=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, stride=stride, padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        self.act = nn.ReLU(inplace=True) if act else nn.Identity()
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class ETM(nn.Module):
    def __init__(self, cin, c):
        super().__init__()
        self.b0 = BasicConv(cin, c, 1)
        self.b1 = nn.Sequential(BasicConv(cin,c,1), BasicConv(c,c,(1,3),padding=(0,1)), BasicConv(c,c,(3,1),padding=(1,0)), BasicConv(c,c,3,padding=3,dilation=3))
        self.b2 = nn.Sequential(BasicConv(cin,c,1), BasicConv(c,c,(1,5),padding=(0,2)), BasicConv(c,c,(5,1),padding=(2,0)), BasicConv(c,c,3,padding=5,dilation=5))
        self.b3 = nn.Sequential(BasicConv(cin,c,1), BasicConv(c,c,(1,7),padding=(0,3)), BasicConv(c,c,(7,1),padding=(3,0)), BasicConv(c,c,3,padding=7,dilation=7))
        self.cat = BasicConv(c*4,c,3,padding=1)
        self.res = BasicConv(cin,c,1)
    def forward(self,x):
        y = self.cat(torch.cat([self.b0(x),self.b1(x),self.b2(x),self.b3(x)],1))
        return F.relu(y+self.res(x), inplace=True)

class DynamicPos(nn.Module):
    def __init__(self, dim):
        super().__init__(); self.proj=nn.Linear(4,dim)
    def forward(self,x):
        b,h,w,c=x.shape; dev=x.device
        yy=torch.linspace(0,1,h,device=dev); xx=torch.linspace(0,1,w,device=dev)
        gy,gx=torch.meshgrid(yy,xx,indexing='ij')
        g=torch.stack([gy,gx,1-gy,1-gx],-1).unsqueeze(0)
        return x + self.proj(g).to(dtype=x.dtype)

class SlotAttention(nn.Module):
    def __init__(self, slots, dim, iters=3, eps=1e-8):
        super().__init__(); self.slots=slots; self.iters=iters; self.eps=eps; self.scale=dim**-0.5
        self.norm_x=nn.LayerNorm(dim); self.norm_s=nn.LayerNorm(dim); self.norm_ff=nn.LayerNorm(dim)
        self.embed=nn.Embedding(slots,dim); self.q=nn.Linear(dim,dim); self.k=nn.Linear(dim,dim); self.v=nn.Linear(dim,dim)
        self.gru=nn.GRUCell(dim,dim)
        self.ff=nn.Sequential(nn.Linear(dim,dim),nn.ReLU(inplace=True),nn.Linear(dim,dim))
    def forward(self,x):
        x=self.norm_x(x); k=self.k(x); v=self.v(x); b,_,d=x.shape
        ids=torch.arange(self.slots,device=x.device)[None].expand(b,-1); s=self.embed(ids)
        for _ in range(self.iters):
            old=s; q=self.q(self.norm_s(s)); a=torch.einsum('bkd,bnd->bkn',q,k)*self.scale
            a=a.softmax(1)+self.eps; a=a/a.sum(-1,keepdim=True).clamp_min(self.eps)
            u=torch.einsum('bkn,bnd->bkd',a,v)
            s=self.gru(u.reshape(-1,d),old.reshape(-1,d)).reshape(b,self.slots,d)
            s=s+self.ff(self.norm_ff(s))
        return s

class SlotGrouping(nn.Module):
    def __init__(self, dim, slots, iters=3):
        super().__init__(); self.dim=dim; self.slots=slots
        self.epos=DynamicPos(dim); self.dpos=DynamicPos(dim); self.norm=nn.LayerNorm(dim)
        self.mlp=nn.Sequential(nn.Linear(dim,dim),nn.ReLU(inplace=True),nn.Linear(dim,dim))
        self.sa=SlotAttention(slots,dim,iters); self.fuse=nn.Conv2d(dim*slots,dim,1,bias=False)
    def forward(self,f):
        b,c,h,w=f.shape
        x=self.epos(f.permute(0,2,3,1)).reshape(b,h*w,c); x=self.mlp(self.norm(x)); s=self.sa(x)
        y=s[:,:,None,None,:].expand(b,self.slots,h,w,c).reshape(b*self.slots,h,w,c)
        y=self.dpos(y).permute(0,3,1,2).reshape(b,self.slots*c,h,w)
        return self.fuse(y)

class AlphaFusion(nn.Module):
    def __init__(self,c):
        super().__init__(); self.f1=nn.Conv2d(c*2,c,1,bias=False); self.f2=nn.Conv2d(c,1,1,bias=False)
    def path(self,x): return self.f2(F.relu(self.f1(x),inplace=True))
    def forward(self,x): return torch.sigmoid(self.path(F.adaptive_avg_pool2d(x,1))+self.path(F.adaptive_max_pool2d(x,1)))

class MFG(nn.Module):
    def __init__(self,c,fine_slots=4,coarse_slots=2,iters=3):
        super().__init__(); self.fine=SlotGrouping(c,fine_slots,iters); self.coarse=SlotGrouping(c,coarse_slots,iters); self.alpha=AlphaFusion(c)
    def forward(self,x):
        f=self.fine(x); c=self.coarse(x+f); a=self.alpha(torch.cat([f,c],1)); return x+a*f+(1-a)*c

class ChannelAttention(nn.Module):
    def __init__(self,c,r=16):
        super().__init__(); h=max(c//r,4); self.m=nn.Sequential(nn.Conv2d(c,h,1,bias=False),nn.ReLU(inplace=True),nn.Conv2d(h,c,1,bias=False))
    def forward(self,x): return torch.sigmoid(self.m(F.adaptive_avg_pool2d(x,1))+self.m(F.adaptive_max_pool2d(x,1)))
class SpatialAttention(nn.Module):
    def __init__(self): super().__init__(); self.c=nn.Conv2d(2,1,7,padding=3,bias=False)
    def forward(self,x): return torch.sigmoid(self.c(torch.cat([x.mean(1,keepdim=True),x.max(1,keepdim=True).values],1)))
class CBAM(nn.Module):
    def __init__(self,c): super().__init__(); self.ca=ChannelAttention(c); self.sa=SpatialAttention()
    def forward(self,x): x=self.ca(x)*x; return self.sa(x)*x
class InterFA(nn.Module):
    def __init__(self,c):
        super().__init__(); self.c3=BasicConv(2*c,c,3,padding=1); self.cb=CBAM(c); self.c1=BasicConv(2*c,c,1)
    def forward(self,shallow,deep):
        deep=F.interpolate(deep,size=shallow.shape[-2:],mode='bilinear',align_corners=False)
        f=self.cb(self.c3(torch.cat([shallow,deep],1))); out=self.c1(torch.cat([f,shallow],1)); return f,out

class GlobalPrior(nn.Module):
    def __init__(self,cin,d=128):
        super().__init__(); ds=(6,12,18)
        self.pool=nn.Sequential(nn.AdaptiveAvgPool2d(1),BasicConv(cin,d,1)); self.b0=BasicConv(cin,d,1)
        self.b1=BasicConv(cin,d,3,padding=ds[0],dilation=ds[0]); self.b2=BasicConv(cin,d,3,padding=ds[1],dilation=ds[1]); self.b3=BasicConv(cin,d,3,padding=ds[2],dilation=ds[2])
        self.head=nn.Sequential(BasicConv(d*5,256,3,padding=1),nn.Conv2d(256,64,3,padding=1,bias=False),nn.BatchNorm2d(64),nn.PReLU(),nn.Dropout2d(.1),nn.Conv2d(64,1,1))
    def forward(self,x):
        sz=x.shape[-2:]; p=F.interpolate(self.pool(x),size=sz,mode='bilinear',align_corners=False)
        return self.head(torch.cat([p,self.b0(x),self.b1(x),self.b2(x),self.b3(x)],1))

class ChannelGate(nn.Module):
    def __init__(self,c,r=16):
        super().__init__(); h=max(c//r,4); self.m=nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Conv2d(c,h,1),nn.ReLU(inplace=True),nn.Conv2d(h,c,1),nn.Sigmoid())
    def forward(self,x): return x*self.m(x)
class RCAB(nn.Module):
    def __init__(self,c): super().__init__(); self.m=nn.Sequential(nn.Conv2d(c,c,3,padding=1),nn.ReLU(inplace=True),nn.Conv2d(c,c,3,padding=1),ChannelGate(c))
    def forward(self,x): return x+self.m(x)
class PriorDecoder(nn.Module):
    def __init__(self,c):
        super().__init__(); self.fg=RCAB(c); self.bg=RCAB(c); self.fuse=BasicConv(2*c,c,3,padding=1); self.pred=nn.Conv2d(c,1,3,padding=1)
    def forward(self,f,prior):
        p=torch.sigmoid(F.interpolate(prior,size=f.shape[-2:],mode='bilinear',align_corners=False)); rp=1-p
        a=self.fg(f*p+f); b=self.bg(f*rp+f); return self.pred(self.fuse(torch.cat([a,b],1)))

class PvtV2B4_WSSAM(nn.Module):
    def __init__(self,pretrained=True,input_norm=True,channels=64,use_checkpoint=False,mfg_enable=True,mfg_fine_slots=4,mfg_coarse_slots=2,mfg_iters=3,deep_supervision=False,**kwargs):
        super().__init__(); del kwargs
        self.encoder=pvt_v2_eff_b4(pretrained=pretrained,use_checkpoint=use_checkpoint); self.embed_dims=list(self.encoder.embed_dims)
        self.normalizer=PixelNormalizer() if input_norm else nn.Identity(); self.deep_supervision=bool(deep_supervision)
        self.e2=ETM(self.embed_dims[0],channels); self.e3=ETM(self.embed_dims[1],channels); self.e4=ETM(self.embed_dims[2],channels); self.e5=ETM(self.embed_dims[3],channels)
        self.mfg=MFG(channels,mfg_fine_slots,mfg_coarse_slots,mfg_iters) if mfg_enable else nn.Identity()
        self.i4=InterFA(channels); self.i3=InterFA(channels); self.i2=InterFA(channels)
        self.prior=GlobalPrior(self.embed_dims[3]); self.d5=PriorDecoder(channels); self.d4=PriorDecoder(channels); self.d3=PriorDecoder(channels); self.d2=PriorDecoder(channels)
    def enc(self,image):
        f=self.encoder(self.normalizer(image)); return f['reduction_2'],f['reduction_3'],f['reduction_4'],f['reduction_5']
    @staticmethod
    def up(x,sz): return F.interpolate(x,size=sz,mode='bilinear',align_corners=False)
    def body_all(self,data):
        image=data['image_m']; c2,c3,c4,c5=self.enc(image); f2,f3,f4,f5=self.e2(c2),self.e3(c3),self.e4(c4),self.mfg(self.e5(c5))
        m4,f4=self.i4(f4,f5); m3,f3=self.i3(f3,m4); _,f2=self.i2(f2,m3)
        pr=self.prior(c5); p5=self.d5(f5,pr); p4=self.d4(f4,p5); p3=self.d3(f3,p4); p2=self.d2(f2,p3); sz=image.shape[-2:]
        return {k:self.up(v,sz) for k,v in {'prior':pr,'p5':p5,'p4':p4,'p3':p3,'p2':p2}.items()}
    def body(self,data): return self.body_all(data)['p2']
    @staticmethod
    def bce(x,y): return F.binary_cross_entropy_with_logits(x,y,reduction='mean')
    def forward(self,data,iter_percentage=1.0,**kwargs):
        del iter_percentage,kwargs; o=self.body_all(data); logits=o['p2']
        if not self.training: return logits
        y=data['mask'].float(); y=y.unsqueeze(1) if y.ndim==3 else y
        if y.shape[-2:]!=logits.shape[-2:]: y=F.interpolate(y,size=logits.shape[-2:],mode='nearest')
        final=self.bce(logits,y); deep=logits.new_zeros(())
        if self.deep_supervision:
            deep=.0625*self.bce(o['prior'],y)+.125*self.bce(o['p5'],y)+.25*self.bce(o['p4'],y)+.5*self.bce(o['p3'],y)
        total=final+deep
        return {'logits':logits,'loss':total,'loss_items':{'total':total.detach(),'bce':final.detach(),'deep':deep.detach()},'loss_str':f'L:{total.detach().item():.4f} BCE:{final.detach().item():.4f} Deep:{deep.detach().item():.4f}','vis':{'sal':logits.sigmoid(),'prior':o['prior'].sigmoid(),'p5':o['p5'].sigmoid(),'p4':o['p4'].sigmoid(),'p3':o['p3'].sigmoid()}}
    def get_grouped_params(self):
        d={'pretrained':[],'fixed':[],'retrained':[]}
        for n,p in self.named_parameters():
            if n.startswith('encoder.patch_embed1.'):
                p.requires_grad=False; d['fixed'].append(p)
            elif n.startswith('encoder.'): d['pretrained'].append(p)
            else: d['retrained'].append(p)
        LOGGER.info(f"Parameter Groups: Pretrained={len(d['pretrained'])}, Fixed={len(d['fixed'])}, Retrained={len(d['retrained'])}")
        return d

__all__=['PvtV2B4_WSSAM']
