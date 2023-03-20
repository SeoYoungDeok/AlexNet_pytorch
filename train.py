import yaml
import torch
from tqdm import tqdm
import data_loader.data_loader as module_data
import model.model as module_model
import loss.loss as module_loss


def main(config):
    device = config["device"]
    epochs = config["epoch"]

    data_loader = getattr(module_data, config["data_loader"])(
        path=config["data_path"], batch=config["batch_size"]
    )
    model = getattr(module_model, config["model"])().to(device)
    loss_fn = getattr(module_loss, config["loss"])
    optimizer = getattr(torch.optim, config["optimizer"])(
        model.parameters(), lr=config["lr"]
    )

    for epoch in range(epochs):
        with tqdm(data_loader) as pbar:
            pbar.set_description(f"Epoch : {epoch}")
            for imgs, labels in pbar:
                imgs = imgs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                pred = model(imgs)
                _, pred_idx = torch.max(pred, dim=1)
                loss = loss_fn(pred, labels)

                loss.backward()
                optimizer.step()

                pbar.set_postfix(
                    loss=f"{loss.item() / imgs.shape[0]:.3f}",
                    acc=f"{torch.sum(pred_idx == labels.data).item() / imgs.shape[0]:.3f}",
                )


if __name__ == "__main__":
    with open("config.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    main(config)
