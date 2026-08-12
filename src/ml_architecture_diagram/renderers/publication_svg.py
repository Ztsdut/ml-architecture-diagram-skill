from __future__ import annotations

from html import escape
from pathlib import Path
import math

from ..layout import layout_figure
from ..routing import fanin_bundle_geometry, fanout_bundle_geometry
from ..scientific_illustration import svg_illustration, illustration_region
from ..theme import theme_for_spec
from ..publication_design import compile_publication_spec


def _wrap(text: str, max_chars: int) -> list[str]:
    # Scientific labels often contain long hyphenated compounds.  Treat hyphens and
    # slashes as legal wrap opportunities before resorting to a hard character split.
    raw = str(text).split()
    words=[]
    for word in raw:
        if len(word) <= max_chars:
            words.append(word); continue
        parts=[]; buf=""
        for ch in word:
            buf += ch
            if ch in {"-", "/"} and len(buf) >= max(4, max_chars//2):
                parts.append(buf); buf=""
        if buf: parts.append(buf)
        if len(parts)==1 and len(parts[0])>max_chars:
            token=parts[0]; parts=[token[i:i+max_chars] for i in range(0,len(token),max_chars)]
        words.extend(parts)
    if not words: return [""]
    lines=[]; cur=""
    for word in words:
        trial=(cur+" "+word).strip()
        if cur and len(trial)>max_chars:
            lines.append(cur); cur=word
        else: cur=trial
    if cur: lines.append(cur)
    if len(lines)>2:
        # Preserve readability instead of silently overflowing: combine the tail only
        # when it fits; otherwise the renderer keeps the first two semantic chunks.
        tail=' '.join(lines[1:])
        lines=[lines[0],tail if len(tail)<=max_chars+3 else lines[1]]
    return lines[:2]


def _text(x,y,text,font,size=11.0,weight=500,anchor="middle",fill="#1F2933",opacity=1.0,italic=False):
    style=' font-style="italic"' if italic else ''
    return f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" font-family="{font}" font-size="{size:.1f}" font-weight="{weight}" fill="{fill}" opacity="{opacity:.2f}"{style}>{escape(str(text))}</text>'


def _m(theme: dict, key: str, default: float) -> float:
    return float(theme.get(key, default))


def _node_label(node, box, theme, font, y=None, size=None, weight=600, shape_below=True):
    x,y0,w,h=box['x'],box['y'],box['w'],box['h']
    size = _m(theme, '_font_main', 18.0) if size is None else float(size)
    yy=y if y is not None else y0+h+size
    max_chars=max(9,int(w/max(size*.56,1)))
    lines=_wrap(node.get('label',''), max_chars)
    out=[]
    gap=size*.92
    if len(lines)==1:
        out.append(_text(x+w/2,yy,lines[0],font,size,weight,fill=theme['text']))
        bottom=yy
    else:
        out.append(_text(x+w/2,yy-gap*.48,lines[0],font,size,weight,fill=theme['text']))
        out.append(_text(x+w/2,yy+gap*.52,lines[1],font,size,weight,fill=theme['text']))
        bottom=yy+gap*.52
    shape=node.get('shape') or node.get('subtitle')
    if shape_below and shape:
        out.append(_text(x+w/2,bottom+_m(theme,'_font_secondary',14.0)*1.15,shape,font,_m(theme,'_font_secondary',14.0),400,fill=theme['text'],opacity=.68))
    return '\n'.join(out)


def _accent(role, theme):
    return theme.get(role, theme['neutral'])


def _annotation(node: dict) -> str:
    """Return one compact paper-facing annotation line for a node."""
    return (_annotations(node) or [''])[0]


def _annotations(node: dict) -> list[str]:
    vals=[]
    for key in ('subtitle','shape'):
        value=node.get(key)
        if value and str(value) not in vals:
            vals.append(str(value))
    details=node.get('details') or []
    if isinstance(details,str):
        details=[details]
    if not vals and details:
        vals.append('; '.join(str(x) for x in details[:2]).replace('also uses raw ', 'raw: '))
    return vals[:2]


def _module(node, box, theme, font, strong=False):
    x,y,w,h=box['x'],box['y'],box['w'],box['h']; role=node.get('role','neutral')
    accent=_accent(role,theme); stroke=theme['stroke']
    sw=1.25 if strong else 1.0
    out=[f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="3" fill="#FFFFFF" stroke="{stroke}" stroke-width="{max(sw,_m(theme,"_stroke_aux",1.5)):.2f}"/>']
    out.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="7" rx="3" fill="{accent}" stroke="none"/>')
    out.append(f'<rect x="{x:.1f}" y="{y+4:.1f}" width="{w:.1f}" height="3" fill="{accent}" stroke="none"/>')
    ill=node.get('illustration') or None
    fs=_m(theme,'_font_main',18.0); afs=_m(theme,'_font_secondary',14.0)
    anns=_annotations(node)
    if ill:
        ix,iy,iw,ih=illustration_region(box,ill)
        out.append(svg_illustration(ill,ix,iy,iw,ih,theme,font))
        comp=str(ill.get('composition','illustration-top'))
        if comp=='illustration-left':
            tx=x+w*.43; tw=w*.52
            lines=_wrap(node.get('label',''), max(9,int(tw/max(fs*.53,1))))
            cy=y+h*.40
            for i,line in enumerate(lines):
                out.append(_text(tx,cy+(i-(len(lines)-1)/2)*fs*.92,line,font,fs,650,anchor='start',fill=theme['text']))
            if anns:
                out.append(_text(tx,y+h*.73,_wrap(anns[0],max(10,int(tw/max(afs*.50,1))))[0],font,afs,400,anchor='start',fill=theme['text'],opacity=.68))
        elif comp=='illustration-right':
            tx=x+w*.05; tw=w*.52
            lines=_wrap(node.get('label',''), max(9,int(tw/max(fs*.53,1))))
            cy=y+h*.40
            for i,line in enumerate(lines):
                out.append(_text(tx,cy+(i-(len(lines)-1)/2)*fs*.92,line,font,fs,650,anchor='start',fill=theme['text']))
            if anns:
                out.append(_text(tx,y+h*.73,_wrap(anns[0],max(10,int(tw/max(afs*.50,1))))[0],font,afs,400,anchor='start',fill=theme['text'],opacity=.68))
        else:
            lines=_wrap(node.get('label',''), max(10,int(w/max(fs*.55,1))))
            label_y=y+h*(.72 if comp!='illustration-bottom' else .30)
            for i,line in enumerate(lines):
                out.append(_text(x+w/2,label_y+(i-(len(lines)-1)/2)*fs*.90,line,font,fs,650,fill=theme['text']))
            if anns:
                ann_y=y+h*(.92 if comp!='illustration-bottom' else .44)
                out.append(_text(x+w/2,ann_y,_wrap(anns[0],max(12,int(w/max(afs*.50,1))))[0],font,afs,400,fill=theme['text'],opacity=.68))
    else:
        lines=_wrap(node.get('label',''), max(10,int(w/max(fs*.56,1))))
        cy=y+h*(.36 if len(anns)>1 else (.43 if anns else .51))
        for i,line in enumerate(lines):
            dy=(i-(len(lines)-1)/2)*(fs*.95)
            out.append(_text(x+w/2,cy+dy,line,font,fs,650,fill=theme['text']))
        if anns:
            base=y+h-10-afs*.92*(len(anns)-1)
            for j,ann in enumerate(anns):
                line_txt=_wrap(ann,max(12,int(w/max(afs*.52,1))))[0]
                out.append(_text(x+w/2,base+j*afs*.92,line_txt,font,afs,400,fill=theme['text'],opacity=.68))
    repeat=int(node.get('repeat',1) or 1)
    if repeat>1:
        out.append(_text(x+w-8,y+h-8,f'×{repeat}',font,_m(theme,'_font_repeat',13.5),650,anchor='end',fill=theme['text'],opacity=.78))
    return '\n'.join(out)

def _input_text_block(node, box, theme, font):
    x,y,w,h=box['x'],box['y'],box['w'],box['h']
    fs=_m(theme,'_font_main',18.0); afs=_m(theme,'_font_secondary',14.0)
    lines=_wrap(node.get('label',''), max(10,int(w/max(fs*.55,1))))
    shape=str(node.get('shape') or node.get('subtitle') or '')
    # Anchor the text block to the bottom so two-line names and tensor shapes never
    # collide with the illustrative glyph above them.
    bottom=y+h-4
    if shape:
        out=[_text(x+w/2,bottom,_wrap(shape,28)[0],font,afs,400,fill=theme['text'],opacity=.68)]
        label_bottom=bottom-afs*1.25
    else:
        out=[]; label_bottom=bottom
    gap=fs*.88
    first=label_bottom-gap*(len(lines)-1)
    for i,line in enumerate(lines):
        out.append(_text(x+w/2,first+i*gap,line,font,fs,620,fill=theme['text']))
    return '\n'.join(out)


def _tensor(node, box, theme, font, output=False):
    x,y,w,h=box['x'],box['y'],box['w'],box['h']; v=node.get('visual') or {}
    scale=float(v.get('spatial_scale',1.0)); ch=float(v.get('channel_scale',1.0))
    accent=_accent('output' if output else node.get('role','input'),theme); stroke=theme['stroke']
    # Geometry has visible scale variation but stays inside the node bounding box.
    face_w=w*(.54*scale + .18); face_h=h*(.42*scale + .17); depth=min(22.0, 9.0+8.0*ch)
    bx=x+(w-face_w-depth)/2; by=y+8+(h*.56-face_h)/2
    out=[]
    layers=4
    for i in range(layers-1,-1,-1):
        dx=i*depth/(layers+1); dy=-i*depth/(layers+1)
        opacity=.34+.14*(layers-i)
        out.append(f'<rect x="{bx+dx:.1f}" y="{by+dy:.1f}" width="{face_w:.1f}" height="{face_h:.1f}" fill="{accent}" fill-opacity="{opacity:.2f}" stroke="{stroke}" stroke-width="{_m(theme,'_stroke_aux',1.5):.2f}"/>')
    # subtle spatial grid only on the front face
    fx,fy=bx,by
    for q in (.33,.66):
        out.append(f'<line x1="{fx+face_w*q:.1f}" y1="{fy:.1f}" x2="{fx+face_w*q:.1f}" y2="{fy+face_h:.1f}" stroke="{stroke}" stroke-width="{_m(theme,'_stroke_aux',1.5)*.45:.2f}" opacity=".23"/>')
        out.append(f'<line x1="{fx:.1f}" y1="{fy+face_h*q:.1f}" x2="{fx+face_w:.1f}" y2="{fy+face_h*q:.1f}" stroke="{stroke}" stroke-width="{_m(theme,'_stroke_aux',1.5)*.45:.2f}" opacity=".23"/>')
    out.append(_input_text_block(node,box,theme,font))
    return '\n'.join(out)


def _feature_vector(node,box,theme,font):
    x,y,w,h=box['x'],box['y'],box['w'],box['h']; accent=_accent(node.get('role','input'),theme); stroke=theme['stroke']
    out=[]; n=8; bw=(w-30-(n-1)*4)/n
    for i in range(n):
        bh=18+((i*7)%20)
        out.append(f'<rect x="{x+15+i*(bw+4):.1f}" y="{y+12+(38-bh):.1f}" width="{bw:.1f}" height="{bh:.1f}" fill="{accent}" stroke="{stroke}" stroke-width="{_m(theme,'_stroke_aux',1.5)*.72:.2f}"/>')
    out.append(_input_text_block(node,box,theme,font))
    return '\n'.join(out)


def _tokens(node,box,theme,font):
    x,y,w,h=box['x'],box['y'],box['w'],box['h']; accent=_accent(node.get('role','input'),theme); stroke=theme['stroke']
    out=[]; n=7; gap=4; bw=(w-24-gap*(n-1))/n; by=y+12
    for i in range(n):
        alpha=.52+.05*(i%3)
        out.append(f'<rect x="{x+12+i*(bw+gap):.1f}" y="{by:.1f}" width="{bw:.1f}" height="31" rx="2" fill="{accent}" fill-opacity="{alpha:.2f}" stroke="{stroke}" stroke-width="{_m(theme,'_stroke_aux',1.5)*.78:.2f}"/>')
    out.append(_input_text_block(node,box,theme,font))
    return '\n'.join(out)


def _input_card(node,box,theme,font):
    x,y,w,h=box['x'],box['y'],box['w'],box['h']; stroke=theme['stroke']; accent=_accent('input',theme)
    out=[f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="5" fill="#FFFFFF" stroke="{stroke}" stroke-width="{_m(theme,"_stroke_aux",1.5):.2f}"/>']
    ill=node.get('illustration') or None
    if ill:
        # Input cards reserve the left third for a semantically meaningful mini-illustration.
        local=dict(ill); local.setdefault('composition','illustration-left')
        ix,iy,iw,ih=illustration_region(box,local)
        out.append(svg_illustration(local,ix,iy,iw,ih,theme,font))
    else:
        vx=x+18; vcy=y+h/2-4; symbol=(node.get('visual') or {}).get('input_symbol','vector')
        if symbol=='tensor':
            for i in range(3,0,-1):
                out.append(f'<rect x="{vx+i*4:.1f}" y="{vcy-23-i*3:.1f}" width="54" height="44" fill="{accent}" fill-opacity="{.20+.10*i:.2f}" stroke="{stroke}" stroke-width="{_m(theme,"_stroke_aux",1.5)*.60:.2f}"/>')
        elif symbol=='tokens':
            for i in range(5):
                out.append(f'<rect x="{vx+i*11:.1f}" y="{vcy-15:.1f}" width="8" height="30" rx="1.5" fill="{accent}" fill-opacity=".45" stroke="{stroke}" stroke-width=".65"/>')
        else:
            heights=[13,28,20,34,17]
            for i,bh in enumerate(heights):
                out.append(f'<rect x="{vx+i*11:.1f}" y="{vcy+17-bh:.1f}" width="7" height="{bh}" fill="{accent}" fill-opacity=".48" stroke="{stroke}" stroke-width=".55"/>')
    tx=x+92; tw=w-104; fs=_m(theme,'_font_small',16.0); afs=_m(theme,'_font_secondary',14.0)
    lines=_wrap(node.get('label',''), max(10,int(tw/max(fs*.54,1))))
    center=y+h*.40
    for i,line in enumerate(lines):
        out.append(_text(tx,center+(i-(len(lines)-1)/2)*fs*.88,line,font,fs,650,anchor='start',fill=theme['text']))
    annotations=_annotations(node)
    if annotations:
        base=y+h*.67
        for j,ann in enumerate(annotations[:2]):
            line_txt=_wrap(ann,max(12,int(tw/max(afs*.50,1))))[0]
            out.append(_text(tx,base+j*afs*.96,line_txt,font,afs,400,anchor='start',fill=theme['text'],opacity=.68))
    return '\n'.join(out)

def _transformer_macro(node,box,theme,font):
    x,y,w,h=box['x'],box['y'],box['w'],box['h']; accent=_accent('novel',theme); stroke=theme['stroke']
    out=[]
    # Layer stack behind the active block.
    for i in range(3,0,-1):
        out.append(f'<rect x="{x+i*5:.1f}" y="{y-i*4:.1f}" width="{w-10:.1f}" height="{h-20:.1f}" rx="3" fill="#FFFFFF" stroke="{stroke}" stroke-width="{_m(theme,'_stroke_aux',1.5)*.82:.2f}" opacity="{.42+.12*i:.2f}"/>')
    out.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w-10:.1f}" height="{h-20:.1f}" rx="3" fill="#FFFFFF" stroke="{stroke}" stroke-width="{_m(theme,'_stroke_main',2.0):.2f}"/>')
    # Two semantic sub-stages without inventing implementation labels.
    yy=y+16; inner_w=w-38
    for j in range(2):
        out.append(f'<rect x="{x+14:.1f}" y="{yy+j*38:.1f}" width="{inner_w:.1f}" height="27" rx="2" fill="{accent}" fill-opacity="{.28 if j==0 else .16}" stroke="{stroke}" stroke-width="{_m(theme,'_stroke_aux',1.5)*.72:.2f}"/>')
        out.append(f'<line x1="{x+24:.1f}" y1="{yy+13+j*38:.1f}" x2="{x+14+inner_w-10:.1f}" y2="{yy+13+j*38:.1f}" stroke="{stroke}" stroke-width="{_m(theme,'_stroke_aux',1.5)*.55:.2f}" opacity=".35"/>')
    fs=_m(theme,'_font_small',16.0)
    lines=_wrap(node.get('label','Transformer block'), max(10,int((w-18)/max(fs*.55,1))))
    base=y+h-31 if len(lines)==1 else y+h-40
    for i,line in enumerate(lines):
        out.append(_text(x+(w-10)/2,base+i*fs*.92,line,font,fs,650,fill=theme['text']))
    repeat=int(node.get('repeat',1) or 1)
    if repeat>1: out.append(_text(x+w-6,y+13,f'×{repeat}',font,_m(theme,'_font_repeat',13.5),700,anchor='end',fill=theme['text']))
    shape=node.get('shape')
    if shape: out.append(_text(x+(w-10)/2,y+h+_m(theme,'_font_secondary',14.0)*.95,shape,font,_m(theme,'_font_secondary',14.0),400,fill=theme['text'],opacity=.64))
    return '\n'.join(out)


def _attention(node,box,theme,font):
    x,y,w,h=box['x'],box['y'],box['w'],box['h']; accent=_accent('novel',theme); stroke=theme['stroke']
    out=[f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="3" fill="#FFFFFF" stroke="{stroke}" stroke-width="{_m(theme,'_stroke_aux',1.5):.2f}"/>']
    # parallel head lanes converging to one projection strip
    n=4; sx=x+18; ex=x+w-18; cy=y+27
    for i in range(n):
        xx=sx+i*(ex-sx)/(n-1)
        out.append(f'<rect x="{xx-7:.1f}" y="{cy-8:.1f}" width="14" height="16" rx="2" fill="{accent}" fill-opacity=".55" stroke="{stroke}" stroke-width=".6"/>')
        out.append(f'<line x1="{xx:.1f}" y1="{cy+8:.1f}" x2="{x+w/2:.1f}" y2="{y+48:.1f}" stroke="{stroke}" stroke-width="{_m(theme,'_stroke_aux',1.5)*.55:.2f}" opacity=".55"/>')
    out.append(f'<rect x="{x+w*.31:.1f}" y="{y+44:.1f}" width="{w*.38:.1f}" height="7" rx="2" fill="{accent}" fill-opacity=".42" stroke="{stroke}" stroke-width=".5"/>')
    fs=_m(theme,'_font_small',16.0); lines=_wrap(node.get('label','Attention'), max(9,int((w-16)/max(fs*.54,1))))
    ann=_annotation(node)
    afs=_m(theme,'_font_secondary',14.0)
    label_bottom=y+h-(afs*1.35 if ann else 8)
    base=label_bottom-(len(lines)-1)*fs*.84
    for i,line in enumerate(lines): out.append(_text(x+w/2,base+i*fs*.84,line,font,fs,650,fill=theme['text']))
    if ann:
        out.append(_text(x+w/2,y+h-5,_wrap(ann,32)[0],font,afs,400,fill=theme['text'],opacity=.68))
    return '\n'.join(out)


def _ffn(node,box,theme,font):
    x,y,w,h=box['x'],box['y'],box['w'],box['h']; accent=_accent(node.get('role','backbone'),theme); stroke=theme['stroke']
    out=[f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="3" fill="#FFFFFF" stroke="{stroke}" stroke-width="{_m(theme,'_stroke_aux',1.5):.2f}"/>']
    widths=[.28,.48,.28]; cx=x+w/2; yb=y+16
    for j in range(2):
        total=w*.72; xx=cx-total/2
        for i,fr in enumerate(widths):
            bw=total*fr/1.04
            out.append(f'<rect x="{xx:.1f}" y="{yb+j*17:.1f}" width="{bw:.1f}" height="8" rx="1.5" fill="{accent}" fill-opacity="{.28+.10*i:.2f}" stroke="none"/>')
            xx+=bw+4
    fs=_m(theme,'_font_small',16.0); lines=_wrap(node.get('label','FFN'), max(9,int((w-12)/max(fs*.54,1))))
    base=y+h-8-(len(lines)-1)*fs*.82
    for i,line in enumerate(lines): out.append(_text(cx,base+i*fs*.82,line,font,fs,650,fill=theme['text']))
    return '\n'.join(out)


def _norm(node,box,theme,font):
    x,y,w,h=box['x'],box['y'],box['w'],box['h']; stroke=theme['stroke']; accent=_accent('preprocess',theme)
    out=[f'<rect x="{x+5:.1f}" y="{y+8:.1f}" width="{w-10:.1f}" height="15" rx="2" fill="{accent}" stroke="{stroke}" stroke-width="{_m(theme,'_stroke_aux',1.5)*.78:.2f}"/>']
    out.append(_text(x+w/2,y+h-8,node.get('label','Norm'),font,_m(theme,'_font_secondary',14.0),600,fill=theme['text']))
    return '\n'.join(out)


def _add(node,box,theme,font):
    x,y,w,h=box['x'],box['y'],box['w'],box['h']; stroke=theme['stroke']; accent=_accent('fusion',theme)
    r=min(w,h)*.34; cx=x+w/2; cy=y+h/2
    return f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" fill="{accent}" fill-opacity=".68" stroke="{stroke}" stroke-width="{_m(theme,'_stroke_aux',1.5):.2f}"/>\n'+_text(cx,cy+4,'+',font,_m(theme,'_font_symbol',21.0),600,fill=theme['text'])


def _sequence_port(node,box,theme,font):
    x,y,w,h=box['x'],box['y'],box['w'],box['h']; stroke=theme['stroke']
    out=[f'<line x1="{x+12:.1f}" y1="{y+h/2:.1f}" x2="{x+w-12:.1f}" y2="{y+h/2:.1f}" stroke="{stroke}" stroke-width="{_m(theme,'_stroke_main',2.0):.2f}"/>']
    out.append(_text(x+w/2,y+13,node.get('label',''),font,_m(theme,'_font_small',16.0),600,fill=theme['text']))
    return '\n'.join(out)


def _fusion(node,box,theme,font):
    x,y,w,h=box['x'],box['y'],box['w'],box['h']; stroke=theme['stroke']; accent=_accent('fusion',theme)
    cx=x+w/2; cy=y+h/2; r=min(w,h)*.28
    symbol='C' if 'concat' in (str(node.get('subtitle',''))+' '+str(node.get('label',''))).lower() else '+'
    out=[f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" fill="{accent}" fill-opacity=".70" stroke="{stroke}" stroke-width="{_m(theme,'_stroke_aux',1.5):.2f}"/>',_text(cx,cy+4,symbol,font,_m(theme,'_font_symbol',21.0),650,fill=theme['text'])]
    out.append(_text(cx,y+h+13,node.get('label','Fusion'),font,_m(theme,'_font_small',16.0),600,fill=theme['text']))
    return '\n'.join(out)


def _fusion_bar(node,box,theme,font):
    x,y,w,h=box['x'],box['y'],box['w'],box['h']; stroke=theme['stroke']; accent=_accent('fusion',theme)
    out=[f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="4" fill="#FFFFFF" stroke="{stroke}" stroke-width="{_m(theme,"_stroke_aux",1.5):.2f}"/>']
    out.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="6" height="{h:.1f}" rx="3" fill="{accent}" stroke="none"/>')
    fs=_m(theme,'_font_small',16.0)
    lines=_wrap(node.get('label','Feature fusion'), max(12,int((w-20)/max(fs*.54,1))))
    cy=y+h*.43
    for i,line in enumerate(lines):
        out.append(_text(x+w/2+3,cy+(i-(len(lines)-1)/2)*fs*.84,line,font,fs,650,fill=theme['text']))
    ann=_annotation(node) or 'Concatenate'
    out.append(_text(x+w/2+3,y+h-7,_wrap(ann,24)[0],font,_m(theme,'_font_secondary',14.0),400,fill=theme['text'],opacity=.66))
    return '\n'.join(out)


def _spectral(node,box,theme,font):
    x,y,w,h=box['x'],box['y'],box['w'],box['h']; stroke=theme['stroke']; accent=_accent('novel',theme)
    out=[f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="3" fill="#FFFFFF" stroke="{stroke}" stroke-width="{_m(theme,'_stroke_main',2.0):.2f}"/>']
    # left: field waveform, center: compact spectrum, right: mixed field. Symbolic only.
    pts=[]
    for i in range(34):
        xx=x+16+(w*.30)*i/33; yy=y+39+9*math.sin(i/33*math.pi*4)
        pts.append(f'{xx:.1f},{yy:.1f}')
    out.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{stroke}" stroke-width="{_m(theme,'_stroke_aux',1.5):.2f}"/>')
    heights=[7,18,30,21,10]
    sx=x+w*.43
    for i,bh in enumerate(heights):
        out.append(f'<rect x="{sx+i*11:.1f}" y="{y+54-bh:.1f}" width="6" height="{bh}" fill="{accent}" fill-opacity=".62" stroke="{stroke}" stroke-width=".45"/>')
    pts=[]
    for i in range(28):
        xx=x+w*.72+(w*.20)*i/27; yy=y+39+6*math.sin(i/27*math.pi*3.2)
        pts.append(f'{xx:.1f},{yy:.1f}')
    out.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{stroke}" stroke-width="{_m(theme,'_stroke_aux',1.5):.2f}"/>')
    out.append(_text(x+w/2,y+h-12,node.get('label','Spectral operator'),font,_m(theme,'_font_main',18.0),650,fill=theme['text']))
    repeat=int(node.get('repeat',1) or 1)
    if repeat>1: out.append(_text(x+w-9,y+15,f'×{repeat}',font,_m(theme,'_font_repeat',13.5),700,anchor='end',fill=theme['text']))
    return '\n'.join(out)


def _linear_map(node,box,theme,font):
    return _module(node,box,theme,font,strong=False)


def _node(node,box,theme,font):
    v=(node.get('visual') or {}).get('type','generic_module')
    if v=='input_card_publication': return _input_card(node,box,theme,font)
    # A verified scientific illustration is a higher-level paper abstraction than
    # a generic visual glyph.  Use it inside the module while preserving the same
    # node identity and topology.
    if node.get('illustration'):
        return _module(node,box,theme,font,strong=node.get('role')=='novel')
    if v in {'input_tensor','feature_tensor','field_tensor'}: return _tensor(node,box,theme,font,False)
    if v=='field_tensor_output': return _tensor(node,box,theme,font,True)
    if v in {'token_strip_publication','token_strip','sequence_strip'}: return _tokens(node,box,theme,font)
    if v=='feature_vector': return _feature_vector(node,box,theme,font)
    if v=='transformer_macro': return _transformer_macro(node,box,theme,font)
    if v=='attention_block': return _attention(node,box,theme,font)
    if v=='ffn_block_publication': return _ffn(node,box,theme,font)
    if v=='norm_bar_publication': return _norm(node,box,theme,font)
    if v=='add_node': return _add(node,box,theme,font)
    if v=='sequence_port': return _sequence_port(node,box,theme,font)
    if v=='fusion_node_publication': return _fusion(node,box,theme,font)
    if v=='fusion_bar_publication': return _fusion_bar(node,box,theme,font)
    if v=='spectral_operator_publication': return _spectral(node,box,theme,font)
    if v in {'linear_map_publication','encoder_module'}: return _module(node,box,theme,font)
    if v=='emphasis_module': return _module(node,box,theme,font,strong=True)
    if v=='pooling_bar_publication': return _module(node,box,theme,font)
    # Existing visual primitives get a restrained paper module rather than decorative glyphs.
    if node.get('kind')=='merge': return _fusion(node,box,theme,font)
    return _module(node,box,theme,font,strong=node.get('role')=='novel')


def _anchor(box, side):
    if side=='left': return box['x'],box['y']+box['h']/2
    if side=='right': return box['x']+box['w'],box['y']+box['h']/2
    if side=='top': return box['x']+box['w']/2,box['y']
    return box['x']+box['w']/2,box['y']+box['h']


def _edge(edge,a,b,theme,font,direction='LR'):
    et=edge.get('type','main'); label=str(edge.get('label',''))
    color=theme['edge']; width=_m(theme,'_stroke_main',2.0) if et=='main' else _m(theme,'_stroke_aux',1.5)
    dash=' stroke-dasharray="5 4"' if et in {'auxiliary','conditioning'} else (' stroke-dasharray="2 4"' if et=='training' else '')
    route=edge.get('_route_points') or []
    if len(route) >= 2 and et != 'residual':
        pts=[(float(p[0]),float(p[1])) for p in route]
        d='M '+f'{pts[0][0]:.1f},{pts[0][1]:.1f}'+' '+ ' '.join(f'L {x:.1f},{y:.1f}' for x,y in pts[1:])
        # Put the label on the longest route segment, which is usually the clean gutter.
        segs=[(abs(x2-x1)+abs(y2-y1),(x1,y1,x2,y2)) for (x1,y1),(x2,y2) in zip(pts,pts[1:])]
        _,(x1,y1,x2,y2)=max(segs,key=lambda z:z[0])
        lx=(x1+x2)/2; ly=(y1+y2)/2-6
    elif et=='residual':
        # Publication residual: compact orthogonal bypass. In vertical Transformer panels use the right side.
        if abs((a['x']+a['w']/2)-(b['x']+b['w']/2)) < 80 and b['y']>a['y']:
            x1,y1=_anchor(a,'right'); x2,y2=_anchor(b,'right'); lane=int(edge.get('_route_lane',0)); rx=max(x1,x2)+40+lane*22
            d=f'M {x1:.1f},{y1:.1f} L {rx:.1f},{y1:.1f} L {rx:.1f},{y2:.1f} L {x2:.1f},{y2:.1f}'
            lx=rx+5; ly=(y1+y2)/2
        else:
            x1,y1=_anchor(a,'top'); x2,y2=_anchor(b,'top'); lane=int(edge.get('_route_lane',0)); ry=min(y1,y2)-28-lane*20
            d=f'M {x1:.1f},{y1:.1f} L {x1:.1f},{ry:.1f} L {x2:.1f},{ry:.1f} L {x2:.1f},{y2:.1f}'
            lx=(x1+x2)/2; ly=ry-5
    elif direction=='LR':
        x1,y1=_anchor(a,'right'); x2,y2=_anchor(b,'left'); mid=(x1+x2)/2
        d=f'M {x1:.1f},{y1:.1f} C {mid:.1f},{y1:.1f} {mid:.1f},{y2:.1f} {x2:.1f},{y2:.1f}'; lx=mid; ly=(y1+y2)/2-8
    else:
        x1,y1=_anchor(a,'bottom'); x2,y2=_anchor(b,'top'); mid=(y1+y2)/2
        d=f'M {x1:.1f},{y1:.1f} C {x1:.1f},{mid:.1f} {x2:.1f},{mid:.1f} {x2:.1f},{y2:.1f}'; lx=(x1+x2)/2; ly=mid
    out=f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width:.2f}" stroke-linecap="round" stroke-linejoin="round" marker-end="url(#arrow)"{dash}/>'
    if label:
        out+='\n'+_text(lx,ly,label,font,_m(theme,'_font_secondary',14.0),500,fill=theme['text'],opacity=.78)
    return out


def _edge_bundle(edges, boxes, target_id, theme, font):
    """Render an adaptive fan-in bus for dense fusion targets."""
    geo=fanin_bundle_geometry(edges,boxes,target_id,offset=22.0)
    if not geo:
        return ''
    color=theme['edge']; sw=_m(theme,'_stroke_aux',1.5); out=[]
    for branch in geo['branches']:
        pts=branch['points']
        if len(pts)<2: continue
        d='M '+f'{pts[0][0]:.1f},{pts[0][1]:.1f}'+' '+ ' '.join(f'L {x:.1f},{y:.1f}' for x,y in pts[1:])
        out.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{sw:.2f}" stroke-linecap="round" stroke-linejoin="round" opacity=".88"/>')
    (x1,y1),(x2,y2)=geo['bus']
    out.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="{sw:.2f}" opacity=".90"/>')
    pts=geo['trunk']; d='M '+f'{pts[0][0]:.1f},{pts[0][1]:.1f}'+' '+ ' '.join(f'L {x:.1f},{y:.1f}' for x,y in pts[1:])
    out.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{_m(theme,"_stroke_main",2.0):.2f}" stroke-linecap="round" marker-end="url(#arrow)"/>')
    return '\n'.join(out)


def _edge_fanout_bundle(edges, boxes, source_id, theme, font):
    """Render one trunk + shared bus + arrowed branches for dense fan-out."""
    geo=fanout_bundle_geometry(edges,boxes,source_id,offset=22.0)
    if not geo:
        return ''
    color=theme['edge']; sw=_m(theme,'_stroke_aux',1.5); out=[]
    pts=geo['trunk']; d='M '+f'{pts[0][0]:.1f},{pts[0][1]:.1f}'+' '+ ' '.join(f'L {x:.1f},{y:.1f}' for x,y in pts[1:])
    out.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{_m(theme,"_stroke_main",2.0):.2f}" stroke-linecap="round"/>')
    (x1,y1),(x2,y2)=geo['bus']
    out.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="{sw:.2f}" opacity=".90"/>')
    for branch in geo['branches']:
        pts=branch['points']
        if len(pts)<2: continue
        d='M '+f'{pts[0][0]:.1f},{pts[0][1]:.1f}'+' '+ ' '.join(f'L {x:.1f},{y:.1f}' for x,y in pts[1:])
        out.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{sw:.2f}" stroke-linecap="round" marker-end="url(#arrow)"/>')
    return '\n'.join(out)


def render_publication_svg(spec: dict, output: str|Path) -> Path:
    output=Path(output); output.parent.mkdir(parents=True,exist_ok=True)
    spec=compile_publication_spec(spec); layout=layout_figure(spec); fig=spec.get('figure',{}); theme=theme_for_spec(spec)
    # Convert manuscript point sizes into SVG user units from the requested final width.
    width_mm=float(fig.get('width_mm',178) or 178); units_per_pt=max(layout['width'],1.0)/width_mm/2.834645669
    theme['_font_main']=7.2*units_per_pt
    theme['_font_small']=6.5*units_per_pt
    theme['_font_secondary']=5.7*units_per_pt
    theme['_font_repeat']=5.9*units_per_pt
    theme['_font_panel']=7.4*units_per_pt
    theme['_font_title']=8.4*units_per_pt
    theme['_font_symbol']=8.0*units_per_pt
    theme['_stroke_main']=0.78*units_per_pt
    theme['_stroke_aux']=0.58*units_per_pt
    font=fig.get('font','Arial')+', Helvetica, sans-serif'; show_title=bool(fig.get('show_title',False))
    top=30.0 if show_title else 6.0
    width=layout['width']; height=layout['height']+top+8
    nodes={n['id']:n for n in spec.get('nodes',[])}
    parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" height="{height:.0f}" viewBox="0 0 {width:.0f} {height:.0f}">',
           f'<defs><marker id="arrow" markerWidth="7" markerHeight="7" refX="6.2" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 z" fill="{theme["edge"]}"/></marker></defs>',
           '<rect x="0" y="0" width="100%" height="100%" fill="#FFFFFF"/>']
    if show_title and fig.get('title'):
        parts.append(_text(width/2,18,fig['title'],font,_m(theme,'_font_title',22.0),650,fill=theme['text']))
    for pl in layout['panels']:
        xo=pl.get('x_offset',0.0); yo=pl.get('y_offset',0.0)+top; p=pl['panel']
        header=(str(p.get('label',''))+' '+str(p.get('title',''))).strip()
        if header:
            parts.append(_text(xo+18,yo+21,header,font,_m(theme,'_font_panel',19.0),650,anchor='start',fill=theme['text']))
        boxes={nid:{**b,'x':b['x']+xo,'y':b['y']+yo} for nid,b in pl['positions'].items()}
        # Complex framework overviews use restrained stage containers; ordinary paper
        # diagrams keep the lighter bracket/header treatment.
        stage_boxes = bool(spec.get('style',{}).get('stage_containers')) and fig.get('layout_preset') == 'publication_framework'
        for gb in pl.get('groups',[]):
            gx=gb['x']+xo; gy=gb['y']+yo; gw=gb['w']; gh=gb['h'];
            if stage_boxes:
                role=str(gb.get('accent','neutral'))
                accent=theme.get(role, theme['panel'])
                heading=theme.get(role+'_text', theme['text'])
                parts.append(f'<rect x="{gx:.1f}" y="{gy:.1f}" width="{gw:.1f}" height="{gh:.1f}" rx="10" fill="{accent}" fill-opacity=".045" stroke="{heading}" stroke-opacity=".34" stroke-width="{_m(theme,'_stroke_aux',1.5)*.92:.2f}"/>')
                parts.append(_text(gx+12,gy+22,gb['label'],font,_m(theme,'_font_panel',19.0),650,anchor='start',fill=heading,opacity=1.0))
            else:
                parts.append(f'<line x1="{gx:.1f}" y1="{gy+8:.1f}" x2="{gx+gw:.1f}" y2="{gy+8:.1f}" stroke="{theme["panel"]}" stroke-width="{_m(theme,'_stroke_aux',1.5)*.78:.2f}"/>')
                parts.append(_text(gx,gy+3,gb['label'],font,_m(theme,'_font_secondary',14.0),600,anchor='start',fill=theme['text'],opacity=.55))
        panel_direction = str(p.get('direction') or fig.get('direction','LR'))
        if fig.get('layout_preset') == 'publication_transformer' and pl['height'] > pl['width']:
            panel_direction = 'TB'
        # Dense fusion nodes use a shared fan-in bus. This is a visual routing
        # transform only; every exact edge remains present in the publication spec.
        incoming_by_target={}
        outgoing_by_source={}
        for e in pl['edges']:
            incoming_by_target.setdefault(e['to'],[]).append(e)
            outgoing_by_source.setdefault(e['from'],[]).append(e)
        bundled=set()
        if fig.get('layout_preset') == 'publication_framework':
            # Fan-in has priority for fusion nodes.
            for tid,ies in incoming_by_target.items():
                node=nodes.get(tid,{})
                vtype=(node.get('visual') or {}).get('type')
                if len(ies)>=3 and vtype in {'fusion_node_publication','fusion_bar_publication'} and all(not e.get('label') for e in ies):
                    parts.append(_edge_bundle(ies,boxes,tid,theme,font))
                    bundled.update((e['from'],e['to'],e.get('label','')) for e in ies)
            # Then compact remaining high-fan-out representation paths.
            for sid,oes in outgoing_by_source.items():
                remaining=[e for e in oes if (e['from'],e['to'],e.get('label','')) not in bundled]
                node=nodes.get(sid,{})
                vtype=(node.get('visual') or {}).get('type')
                if len(remaining)>=3 and vtype in {'encoder_module','emphasis_module','attention_block','feature_vector'} and all(not e.get('label') for e in remaining):
                    drawing=_edge_fanout_bundle(remaining,boxes,sid,theme,font)
                    if drawing:
                        parts.append(drawing)
                        bundled.update((e['from'],e['to'],e.get('label','')) for e in remaining)
        for e in pl['edges']:
            if (e['from'],e['to'],e.get('label','')) in bundled:
                continue
            ee=dict(e)
            if e.get('_route_points'):
                ee['_route_points']=[[float(q[0])+xo,float(q[1])+yo] for q in e['_route_points']]
            parts.append(_edge(ee,boxes[e['from']],boxes[e['to']],theme,font,panel_direction))
        for nid,b in boxes.items():
            node = nodes[nid]
            parts.append(f'<g id="node-{escape(str(nid),quote=True)}">{_node(node,b,theme,font)}</g>')
            ref = node.get("detail_panel_ref")
            if ref:
                parts.append(_text(b["x"]+8, b["y"]+15, str(ref), font, _m(theme,"_font_secondary",14.0), 700, anchor="start", fill=theme.get("novel_text",theme["text"])))
    parts.append('</svg>')
    output.write_text('\n'.join(parts),encoding='utf-8')
    return output
