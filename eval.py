# %%
import os
import torch, pickle
import argparse
from model import FiringRateRNN
from inference import inference


def main():
    parser = argparse.ArgumentParser(description="Run RNN inference and save results")
    parser.add_argument(
        "--model",
        type=str,
        default="/Users/juntao/Desktop/proj_TimePerception/model/"
                "dataset_epoch-40_batchSize-64_initLR-00005_stepSize-15_gamma-09/"
                "[04-30]-[11-48]-fold3-best_model.pt",
        help="Path to the trained model checkpoint (.pt)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="/Users/juntao/Desktop/proj_TimePerception/data/dataset.pt",
        help="Path to the trials_list .pt file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Directory to save inference results"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="inference.pkl",
        help="Filename for saving inference results within output_dir"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run inference on (cpu or cuda)"
    )
    parser.add_argument(
        "--add_noise",
        action="store_true",
        help="If set, add training noise during inference"
    )
    args = parser.parse_args()

    # 1) load model
    device = args.device
    model = FiringRateRNN(hidden_size=200, input_dim=5).to(device)
    # model = FiringRateRNN(hidden_size=200, input_dim=4).to(device)
    ckpt = torch.load(args.model, map_location=device)
    model.load_state_dict(ckpt)

    # 2) load data
    trials_list = torch.load(args.dataset)

    # 3) inference
    all_data = inference(model, trials_list, device=device, add_noise=False) #threshold_value=2.0

    # 4) save
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, args.output_file)
    with open(out_path, "wb") as f:
        pickle.dump(all_data, f)


if __name__ == "__main__":
    main()




