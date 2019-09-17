# plot
if complabels is None:
    complabels = [x['name'] for x in cmpds]

if makefig:
    xlabel = "time / h"
    fig, (ax, ax2) = plot_evap(x=output_dict['t_a'] / 3600,
                               molec_data=evap_a,
                               r_data=r_a,
                               series_labels=complabels,
                               xlabel=xlabel)
    output_dict.update({'evap_fig': (fig, (ax, ax2))})
else:
    output_dict.update({'evap_fig': None})

# e-folding times, converted from seconds to hours
efold_dict = {l: efold_time(output_dict['t_a'] / 3600, evap_a[:, i])
              for i, l in enumerate(complabels)}
output_dict.update({'efold_dict': efold_dict})