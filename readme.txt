Political-Stance-Prediction Folder Structure

├── reader/                     # module to ingest data for training and tevaluation
├── text_cnn/                   # CNN model implementation for text classification
├── scripts
│   ├── __init__.py
│   ├── collect_data.py
│   ├── compare_scores.py
│   ├── corpus_analysis.py
│   ├── fastText_clf.py
│   ├── flair_plot.py
│   ├── generate_csv.py
│   ├── parse_moral_dict.py
│   ├── result2latex.py
│   └── sentence_length_stats.py
├── train_BERT.py               # Fine-tune BERT base
├── train_CNN.py                # Train/test CNN
├── train_ML.py                 # Train/test machine learning models (logistic regression, SVM, MLP)
├── viz_bert/
│   ├── bertviz/
│   └── head_view_bert.ipynb    # Visualize BERT attention weights using bertviz
├── viz_eli5.ipynb              # Visualize machine learning model weights using eli5
└── viz_lime.ipynb              # Visualize CNN weights using lime
