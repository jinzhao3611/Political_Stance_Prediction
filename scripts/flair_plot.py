from flair.visual.training_curves import Plotter

plotter = Plotter()
# plotter.plot_weights('flair_outputs_glove/weights.txt')
# plotter.plot_training_curves('flair_outputs_glove/loss.tsv')
# plotter.plot_learning_rate('flair_outputs_glove/loss.tsv')

plotter.plot_weights("flair_outputs_fastText/weights.txt")
plotter.plot_training_curves("flair_outputs_fastText/loss.tsv")
plotter.plot_learning_rate("flair_outputs_fastText/loss.tsv")
