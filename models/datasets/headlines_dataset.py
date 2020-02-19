from .generic_dataset import GenericDataset
import json


class HeadlinesDataset(GenericDataset):
    def __init__(self, transforms=None):
        super(HeadlinesDataset, self).__init__(transforms)

    @classmethod
    def from_json(cls, json_filename, postprocess_data_func=None,**kwargs):
        with open(json_filename, "r") as json_file:
            content = json_file.read().split('\n')
            examples = []
            for line in content:
                example = json.loads(line)
                examples.append((example["headline"], float(example["is_sarcastic"])))

            if postprocess_data_func is not None:
                for n, e in enumerate(examples):
                    try:
                        e = postprocess_data_func(e)
                        examples[n] = e
                    except ValueError as e:
                        print(e)
                        examples[n] = None

            examples = [e for e in examples if e is not None]
            train, test = cls._standard_creator(examples, **kwargs)

            return train, test
