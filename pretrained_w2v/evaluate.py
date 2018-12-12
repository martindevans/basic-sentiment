import numpy as np
import progressbar as pb

from sklearn.metrics import classification_report

import pretrained_w2v.parameters as pr

def evaluate(dataset, predictor):

    print(" - This machine learning model is really fantastic:")
    print(predictor.sentiment("This machine learning model is really fantastic"))
    print(" - This piece of rubbish is barely worth using:")
    print(predictor.sentiment("This piece of rubbish is barely worth using"))

    actual = []
    expected = []

    val_gen = dataset.val_gen()
    with pb.ProgressBar(widgets=[ pb.Percentage(), ' ', pb.AdaptiveETA(), ' ', pb.Bar() ], max_value=pr.evaluate_batches) as bar:
        for i in range(0, pr.evaluate_batches):
            bar.update(i)
            (input_data, classes) = next(val_gen)
            output_classes = predictor.model.predict_on_batch(input_data)
            actual.extend(map(np.argmax, output_classes))
            expected.extend(map(np.argmax, classes))

    return classification_report(expected, actual, target_names=["neg", "pos", "neut"])