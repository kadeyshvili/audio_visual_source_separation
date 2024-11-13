def update_parameter(model, pretrained_dict):
    model_dict = model.state_dict()
    update_dict = {}
    for k, v in pretrained_dict.items():
        part = k.split('.')[0]
        if part == "frontend3D" or part == "trunk":
            k_ = 'feature_extractor.' + k
            update_dict[k_] = v

    model_dict.update(update_dict)
    model.load_state_dict(model_dict)
    return model


