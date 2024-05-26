from prettytable import PrettyTable


def model_summary(model):
    print()
    print('model_summary')

    table = PrettyTable(["Modules", "Parameters", "Trainable parameters"])
    total_params = 0
    total_trainable_params = 0
    for name, parameter in model.named_parameters():
        params = parameter.numel()
        if not parameter.requires_grad:
            trainable_params = 0
        else:
            trainable_params = params

        table.add_row([name, params, trainable_params])
        total_params += params
        total_trainable_params += trainable_params

    print(table)
    print(f"Total params: {total_params}")
    print(f"Total trainable params: {total_trainable_params}")
    print()
