"""
RC 橋柱剝落高度分析工具 v7
下方文字區：Height: 0.xxx（剝落高度比）
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

FILES = {
    'gen_307':  '/Data/home/vicky/graf260518_im64/results/column20260608_twostage_film_damage_damage_proxy/eval/sr256_RS307_it159999.png',
    'gen_315':  '/Data/home/vicky/graf260518_im64/results/column20260608_twostage_film_damage_damage_proxy/eval/sr256_RS315_it159999.png',
    'gen_330':  '/Data/home/vicky/graf260518_im64/results/column20260608_twostage_film_damage_damage_proxy/eval/sr256_RS330_it159999.png',
    'gen_615':  '/Data/home/vicky/graf260518_im64/results/column20260608_twostage_film_damage_damage_proxy/eval/sr256_RS615_it159999.png',
    'real_307': '/Data/home/vicky/graf260518_im64/RS307.jpg',
    'real_315': '/Data/home/vicky/graf260518_im64/RS315.jpg',
    'real_330': '/Data/home/vicky/graf260518_im64/RS330.jpg',
    'real_615': '/Data/home/vicky/graf260518_im64/RS615.jpg',
}

SPECIMENS    = ['307', '315', '330', '615']
ANGLE_LABELS = ['90°','135°','180°','225°','270°','315°','0°','45°']
PANEL_XS     = [2 + i * (256 + 2) for i in range(8)]
ROW_START, ROW_END = 2, 258

OUT_DIR = '/Data/home/vicky/graf260518_im64/spalling'
os.makedirs(OUT_DIR, exist_ok=True)


def get_font(size):
    for path in [
        '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
        '/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf',
    ]:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    return ImageFont.load_default()


def load_panels(key):
    arr = np.array(Image.open(FILES[key]).convert('RGB'))
    return [arr[ROW_START:ROW_END, xs:xs+256, :] for xs in PANEL_XS]


def detect_spalling_height(panel):
    """剝落高度比（從底部算起，0~1）"""
    h        = 256
    gray     = panel.mean(axis=2)
    sm = np.convolve(gray.mean(axis=1), np.ones(7)/7, mode='same')
    ss = np.convolve(gray.std(axis=1),  np.ones(7)/7, mode='same')
    ref_mean = np.median(sm[:int(h*0.2)])
    ref_std  = np.median(ss[:int(h*0.2)])
    for r in range(int(h*0.15), int(h*0.90)):
        drop = ref_mean - sm[r]
        rise = ss[r]   - ref_std
        if drop > 12 or rise > 12:
            wd = ref_mean - sm[r:min(r+12, h)]
            wr = ss[r:    min(r+12, h)] - ref_std
            if np.sum((wd > 8) | (wr > 8)) >= 7:
                return float((h - r) / h)
    return 0.0


def analyze_all():
    results = {}
    for spec in SPECIMENS:
        gp = load_panels(f'gen_{spec}')
        rp = load_panels(f'real_{spec}')
        results[spec] = {
            'gen':  {'height': [detect_spalling_height(p) for p in gp]},
            'real': {'height': [detect_spalling_height(p) for p in rp]},
        }
    return results


def plot_panels_only(panels, heights, line_color, out_path):
    """
    白色背景，8 panel 橫排。
    偵測線標示剝落高度。
    下方文字區一行：Height: 0.xxx
    """
    pw, ph  = 256, 256
    gap     = 4
    label_h = 44

    W = pw*8 + gap*7
    H = ph + label_h
    canvas = np.ones((H, W, 3), dtype=np.uint8) * 255

    font_val   = get_font(22)
    font_label = get_font(18)
    line_rgb   = (0, 200, 200) if line_color == 'cyan' else (220, 50, 50)

    for i, (panel, h_ratio) in enumerate(zip(panels, heights)):
        x0 = i * (pw + gap)
        canvas[:ph, x0:x0+pw, :] = panel
        if h_ratio > 0.0:
            line_y = int(ph * (1 - h_ratio))
            line_y = max(0, min(line_y, ph-1))
            t = 2
            canvas[max(0,line_y-t):min(ph,line_y+t+1), x0:x0+pw, :] = line_rgb

    img  = Image.fromarray(canvas)
    draw = ImageDraw.Draw(img)

    for i, h_ratio in enumerate(heights):   # 修正：直接 enumerate(heights)
        x0     = i * (pw + gap)
        label  = 'Height:'
        value  = f'{h_ratio:.3f}'
        lb = draw.textbbox((0,0), label, font=font_label)
        vb = draw.textbbox((0,0), value, font=font_val)
        lw = lb[2]-lb[0]; lh_ = lb[3]-lb[1]
        vw = vb[2]-vb[0]; vh_ = vb[3]-vb[1]
        row_h  = max(lh_, vh_)
        row_y  = ph + (label_h - row_h) // 2
        total_w = lw + 4 + vw
        tx = x0 + (pw - total_w) // 2
        draw.text((tx,          row_y + (row_h-lh_)//2), label, font=font_label, fill=(100,100,100))
        draw.text((tx + lw + 4, row_y + (row_h-vh_)//2), value, font=font_val,   fill=(0,0,0))

    img.save(out_path, dpi=(150,150))
    print(f'Saved: {out_path}')


def print_table(results):
    print('\n' + '='*80)
    print(f'{"":8}  {"":6}  ' + '  '.join(f'{a:>7}' for a in ANGLE_LABELS) + '    Mean')
    print('-'*80)
    for spec in SPECIMENS:
        print(f'\nRS{spec}')
        for t in ['gen','real']:
            vals = results[spec][t]['height']
            print(f'  {t:<4} height  ' +
                  '  '.join(f'{v:>7.3f}' for v in vals) +
                  f'  mean={np.mean(vals):.3f}')
        gv  = np.array(results[spec]['gen']['height'])
        rv  = np.array(results[spec]['real']['height'])
        mae = np.mean(np.abs(gv-rv))
        print(f'  MAE  height  ' +
              '  '.join(f'{v:>+7.3f}' for v in gv-rv) +
              f'  MAE={mae:.3f}')


if __name__ == '__main__':
    print('分析中...')
    results = analyze_all()
    print_table(results)

    for spec in SPECIMENS:
        gp = load_panels(f'gen_{spec}')
        rp = load_panels(f'real_{spec}')

        plot_panels_only(gp, results[spec]['gen']['height'],
                         line_color='cyan',
                         out_path=os.path.join(OUT_DIR, f'gen_RS{spec}.png'))

        plot_panels_only(rp, results[spec]['real']['height'],
                         line_color='red',
                         out_path=os.path.join(OUT_DIR, f'real_RS{spec}.png'))

    print('\n全部完成！')