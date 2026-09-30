"""Recreate the post's figures, equations and tables entirely from committed data."""
import argparse
import json
import textwrap
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
SPECIES = ['human', 'mouse', 'mean']


def markdown(headers, rows):
    return '| ' + ' | '.join(headers) + ' |\n|' + '|'.join(['---'] * len(headers)) + '|\n' + '\n'.join('| ' + ' | '.join(map(str, row)) + ' |' for row in rows) + '\n'


def fit_scaling(x, y):
    def predict(p, xx):
        return p[0] - p[1] * xx ** (-p[2])
    fits = [least_squares(lambda p: predict(p, x) - y, [.75, .1, alpha],
                         bounds=([max(y), 0, .001], [1, 10, 5]),
                         max_nfev=20000, ftol=1e-13, xtol=1e-13, gtol=1e-13)
            for alpha in [.05, .1, .3, .8, 1, 2]]
    result = min(fits, key=lambda f: np.sum(f.fun ** 2))
    return result.x, float(np.sqrt(np.mean(result.fun ** 2))), predict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'generated')
    parser.add_argument('--extrapolation-factor', type=float, default=100,
                        help='Extend scaling plots this far beyond the largest observed x (default: 100).')
    args = parser.parse_args()
    if args.extrapolation_factor < 1:
        parser.error('--extrapolation-factor must be >= 1')
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.spines.top': False, 'axes.spines.right': False})
    archived = json.loads((ROOT / 'raw/wandb/archived_analysis_metrics.json').read_text())
    runs = {r['name']: r for r in archived}
    for path in sorted((ROOT / 'raw/wandb/runs').glob('*.json')):
        r = json.loads(path.read_text())
        # Archived post values remain the source of published summaries; fresh full histories are supplemental.
        runs.setdefault(r['name'], r)
    tables = []
    fit_records = {}

    def save(fig, name):
        fig.tight_layout()
        for ext in ['png', 'pdf', 'svg']:
            fig.savefig(out / f'{name}.{ext}', dpi=250)
        plt.close(fig)

    def table(name, headers, rows):
        text = markdown(headers, rows)
        (out / f'{name}.md').write_text(text)
        fig, ax = plt.subplots(figsize=(max(10, len(headers)*1.6), .48*len(rows)+1.5))
        ax.axis('off')
        rendered = ax.table(cellText=rows, colLabels=[textwrap.fill(h, 18) for h in headers], loc='center', cellLoc='center')
        rendered.auto_set_font_size(False)
        rendered.set_fontsize(8 if len(headers)>7 else 9)
        rendered.scale(1, 1.8)
        for (row, col), cell in rendered.get_celld().items():
            cell.set_edgecolor('#d8dee6')
            if row == 0:
                cell.set_height(cell.get_height()*1.8)
                cell.set_facecolor('#24384a')
                cell.set_text_props(color='white', weight='bold')
            elif row % 2:
                cell.set_facecolor('#f0f4f8')
        save(fig, name+'_table')
        tables.append(f'## {name.replace("_", " ").title()}\n\n' + text)

    def scaling(df, name, key, scale, xlabel, formula_x):
        x = df[key].to_numpy(dtype=float) * scale
        y = df['mean_pearson'].to_numpy(dtype=float)
        params, rmse, predict = fit_scaling(x, y)
        ri, amplitude, alpha = params
        fit_records[name] = dict(r_inf=ri, A=amplitude, alpha=alpha, rmse=rmse,
                                form='r_inf - A * x^(-alpha)', x_units=xlabel)
        grid = np.geomspace(min(x), max(x) * args.extrapolation_factor, 800)
        measured = grid <= max(x)
        display_scale = 100 if key == 'fraction' else 1
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(grid[measured] * display_scale, predict(params, grid[measured]), color='#2377a4', label='Power-law fit')
        extension = grid >= max(x)
        ax.plot(grid[extension] * display_scale, predict(params, grid[extension]), '--', color='#2377a4', label='Extrapolation')
        ax.scatter(x * display_scale, y, color='#242d38', s=50, zorder=3, label='Observed species mean')
        ax.axhline(ri, color='#888888', ls=':', label=f'Fitted asymptote ({ri:.3f})')
        ax.set(xscale='log', xlabel=xlabel, ylabel='Test mean Pearson', title=name.replace('_', ' ').capitalize())
        ax.grid(alpha=.15)
        ax.legend(frameon=False)
        save(fig, name)
        formula = rf'\widehat{{\bar r}}={ri:.8f}-{amplitude:.8f}{formula_x}^{{-{alpha:.8f}}}'
        fig = plt.figure(figsize=(10, 1.4))
        fig.text(.5, .5, '$' + formula + '$', ha='center', va='center', fontsize=21)
        save(fig, name + '_equation')
        tables.append(f'$$ {formula}. $$\n\nIn-sample RMSE: {rmse:.6f} Pearson units.\n')
        df.to_csv(out / f'{name}.csv', index=False)

    counts = json.loads((ROOT / 'raw/window_counts.json').read_text())['rows']
    data = []
    for row in counts:
        fraction = row['fraction']
        run = runs[f'basic-model-data{round(100*fraction)}pct']
        data.append(dict(fraction=fraction, paired_examples=row['paired_examples'],
                         unique_human_windows=row['unique_human_windows'], unique_mouse_windows=row['unique_mouse_windows'],
                         **{s + '_pearson': run['summary'][f'test/{s}_pearson'] for s in SPECIES}, source=run['url']))
    data = pd.DataFrame(data)
    table('dataset_scaling', ['Proportion', 'Paired examples', 'Unique human windows*', 'Unique mouse windows*', 'Human test Pearson', 'Mouse test Pearson', 'Mean test Pearson'],
          [[f'{r.fraction*100:g}%', f'{r.paired_examples:,}', f'{r.unique_human_windows:,}', f'{r.unique_mouse_windows:,}', *[f'{getattr(r,s+"_pearson"):.6f}' for s in SPECIES]] for r in data.itertuples()])
    scaling(data, 'dataset_scaling', 'fraction', 1, 'Training subset (%) · log scale', 'f')
    params = []
    for size in ['tiny', 'small', 'medium', 'large', 'full']:
        folder = REPO / 'results' / f'basic-model-parameters-seed42-{size}'
        metrics = json.loads((folder / 'test_results.json').read_text())
        arch = json.loads((folder / 'model_architecture.json').read_text())
        params.append(dict(model=size, parameters=arch['total_parameters'], **{s+'_pearson': metrics[f'test_{s}_pearson'] for s in SPECIES}))
    params = pd.DataFrame(params)
    table('parameter_scaling', ['Model', 'Parameters', 'Human test Pearson', 'Mouse test Pearson', 'Mean test Pearson'],
          [[r.model, f'{r.parameters:,}', *[f'{getattr(r,s+"_pearson"):.6f}' for s in SPECIES]] for r in params.itertuples()])
    scaling(params, 'parameter_scaling', 'parameters', 1e-6, 'Trainable parameters (millions) · log scale', r'\left(P/10^6\right)')

    bench_dir = REPO / 'results/track-comparison-20260919T202732.863735Z'
    bench = pd.read_csv(bench_dir / 'summary.csv').sort_values(['fraction', 'species'])
    # Validate summaries against the raw per-track correlations, using common defined tracks.
    per_track = pd.read_csv(bench_dir / 'per_track.csv')
    for row in bench.itertuples():
        selected = per_track[(per_track['run'] == row.run) & (per_track.species == row.species)]
        selected = selected.dropna(subset=['focused_pearson', 'full_pearson'])
        assert len(selected) > 0
        np.testing.assert_allclose([selected.focused_pearson.mean(), selected.full_pearson.mean()], [row.focused_mean_pearson, row.full_mean_pearson], atol=1e-12)
    bench.to_csv(out / 'track_comparison.csv', index=False)
    table('track_comparison', ['Track fraction', 'Species', 'Selected tracks', 'Focused Pearson', 'Full-model Pearson, same tracks', 'Δ focused − full'],
          [[f'{r.fraction*100:g}%', r.species.capitalize(), r.selected_tracks, f'{r.focused_mean_pearson:.6f}', f'{r.full_mean_pearson:.6f}', f'{r.mean_pearson_delta_focused_minus_full:+.6f}'] for r in bench.itertuples()])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), sharey=True)
    for ax, species in zip(axes, ['human', 'mouse']):
        frame = bench[bench.species == species]
        ax.plot(frame.fraction*100, frame.focused_mean_pearson, 'o-', label='Focused model')
        ax.plot(frame.fraction*100, frame.full_mean_pearson, 's--', label='Full model, same tracks')
        ax.set(xscale='log', xlabel='Selected tracks (%)', title=species.capitalize())
        ax.set_xticks([1, 3, 10, 30], ['1', '3', '10', '30'])
        ax.grid(alpha=.15)
    axes[0].set_ylabel('Mean selected-track test Pearson')
    axes[1].legend(frameon=False)
    save(fig, 'track_comparison')
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for offset, species in [(-.16, 'human'), (.16, 'mouse')]:
        ax.bar(np.arange(4)+offset, bench[bench.species == species].mean_pearson_delta_focused_minus_full, width=.3, label=species.capitalize())
    ax.axhline(0, color='black', lw=1)
    ax.set_xticks(range(4), ['1%', '3%', '10%', '30%'])
    ax.set(xlabel='Selected tracks', ylabel='Pearson difference: focused − full')
    ax.legend(frameon=False)
    save(fig, 'track_deltas')

    depth_specs = [('enformer-human-mouse-baseline-0layer', 'Original', 0, [6000]),
                   ('enformer-human-mouse-baseline-1layer', 'Original', 1, [6000]),
                   ('enformer-human-mouse-baseline-8layer', 'Original', 8, [6000]),
                   ('enformer-human-mouse-baseline-1layer-rope', 'RoPE', 1, [6000, 11000]),
                   ('enformer-human-mouse-baseline-8layer-rope', 'RoPE', 8, [6000, 11000])]
    depth_rows = []
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for name, encoding, blocks, steps in depth_specs:
        r = runs[name]
        history = pd.DataFrame(r.get('validation_history', r.get('history', []))).dropna(subset=['eval/mean_pearson'])
        ax = axes[int(encoding == 'RoPE')]
        ax.plot(history['train/global_step'], history['eval/mean_pearson'], '.-', label=f'{blocks} blocks')
        for step in steps:
            point = history[history['train/global_step'] == step]
            assert len(point) == 1, (name, step)
            point = point.iloc[0]
            depth_rows.append([encoding, blocks, step, *[point[f'eval/{s}_pearson'] for s in SPECIES], r['url']])
    for ax, encoding, step in zip(axes, ['Original positional encoding', 'RoPE'], [6000, 11000]):
        ax.axvline(step, color='gray', ls='--')
        ax.set(title=encoding, xlabel='Optimizer steps', ylabel='Validation mean Pearson')
        ax.grid(alpha=.15)
        ax.legend(frameon=False)
    save(fig, 'transformer_depth')
    for idx, name in enumerate(['original_transformer_depth', 'rope_transformer_depth']):
        fig, ax = plt.subplots(figsize=(8, 4.6))
        for line in axes[idx].lines[:-1]:
            ax.plot(line.get_xdata(), line.get_ydata(), '.-', label=line.get_label())
        ax.set(xlabel='Optimizer steps', ylabel='Validation mean Pearson')
        ax.legend(frameon=False)
        ax.grid(alpha=.15)
        save(fig, name)
    depth_rows.sort(key=lambda r: (r[0] != 'Original', r[2], r[1]))
    pd.DataFrame(depth_rows, columns=['positional_encoding', 'blocks', 'step', 'human_pearson', 'mouse_pearson', 'mean_pearson', 'source']).to_csv(out/'transformer_depth.csv', index=False)
    table('transformer_depth', ['Positional encoding', 'Blocks', 'Matched step', 'Human validation Pearson', 'Mouse validation Pearson', 'Mean validation Pearson'],
          [[enc, blocks, f'{step:,}', *[f'{v:.6f}' for v in [h,m,mean]]] for enc,blocks,step,h,m,mean,url in depth_rows])

    cosine = pd.read_csv(ROOT / 'raw/transformer_usage/all_per_sequence.csv')
    for model, frame in cosine.groupby('model', sort=False):
        pivot = frame.pivot(index='block', columns='sequence_id', values='cosine')
        assert pivot.shape == ({'enformer':11, 'borzoi':8, 'alphagenome':9}[model], 8)
        assert np.isfinite(pivot.to_numpy()).all()
        pivot.to_csv(out/f'{model}_individual_layers.csv')
        table(model+'_individual_layers', ['Block', *pivot.columns], [[i, *[f'{v:.6f}' for v in row]] for i,row in pivot.iterrows()])
        fig, ax = plt.subplots(figsize=(8,4))
        for col in pivot:
            ax.plot(pivot.index, pivot[col], alpha=.3)
        ax.errorbar(pivot.index, pivot.mean(axis=1), yerr=pivot.std(axis=1), color='black', fmt='o-', capsize=3, label='Mean ± sequence SD')
        ax.set(title=model, xlabel='Transformer block', ylabel='Input-output cosine similarity')
        ax.set_xticks(pivot.index)
        ax.grid(alpha=.2)
        ax.legend()
        save(fig, model+'_individual_layers')
    metric_formulas = {
        'pearson': r'r_{s,t}=\frac{\sum_i(y_i-\bar y)(\hat y_i-\overline{\hat y})}{\sqrt{\sum_i(y_i-\bar y)^2\sum_i(\hat y_i-\overline{\hat y})^2}}',
        'mean_pearson': r'\bar r=(\bar r_{\mathrm{human}}+\bar r_{\mathrm{mouse}})/2',
        'matched_track': r'\Delta\bar r_s=\frac{1}{|T_s|}\sum_{t\in T_s}(r^{\mathrm{focused}}_{s,t}-r^{\mathrm{full}}_{s,t})',
        'cosine': r'c_l=\frac{1}{|I|}\sum_{i\in I}\frac{\mathbf{h}_{l,i}\cdot\mathbf{h}_{l+1,i}}{\|\mathbf{h}_{l,i}\|\|\mathbf{h}_{l+1,i}\|}',
    }
    for name, formula in metric_formulas.items():
        fig = plt.figure(figsize=(11,1.5))
        fig.text(.5,.5,'$'+formula+'$',ha='center',va='center',fontsize=22)
        save(fig,name+'_equation')
        tables.append('$$ '+formula+' $$\n')
    (out/'fits.json').write_text(json.dumps(fit_records,indent=2)+'\n')
    (out/'tables-and-equations.md').write_text('# Tables and equations\n\n*Window counts are reconstructed, not original saved subset manifests. Scaling asymptotes are conditional extrapolations; no repeat-seed uncertainty is estimated. Transformer depth uses validation metrics; track comparisons use identical selected tracks with defined correlations in both models. Cosine scores are diagnostic, not measures of layer necessity. See analysis/README.md for provenance and methods.*\n\n'+'\n'.join(tables))
    print(f'Generated figures, tables, equations and source CSVs in {out}')

if __name__ == '__main__':
    main()
