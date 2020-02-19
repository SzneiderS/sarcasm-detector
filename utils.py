def one_hot_vector(length, indices):
    vec = [0 for i in range(0, length)]
    if not isinstance(indices, list):
        indices = [indices]
    for i in indices:
        vec[i] = 1
    return vec


def calculate_one_hot_from_file(filename, process_cb):
    with open(filename, "r") as file:
        classes = list(process_cb(file))
        num_of_classes = len(classes)
        one_hots = {}
        for cls in classes:
            one_hots[cls] = one_hot_vector(num_of_classes, classes.index(cls))
        return one_hots, num_of_classes
