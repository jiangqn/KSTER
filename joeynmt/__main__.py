import argparse

from joeynmt.training import train
from joeynmt.combiner_training import combiner_train
from joeynmt.prediction import test
from joeynmt.prediction import translate
from joeynmt.prediction import analyze

def main():
    ap = argparse.ArgumentParser("KSTER")

    ap.add_argument("mode", choices=["train", "combiner_train", "test", "build_database", "analyze", "translate", "score_translations"],
                    help="train a model or test or translate")

    ap.add_argument("config_path", type=str,
                    help="path to YAML config file")

    ap.add_argument("--ckpt", type=str,
                    help="checkpoint for prediction")

    ap.add_argument("--combiner", type=str, default="no_combiner", 
                    choices=["no_combiner", "static_combiner", "dynamic_combiner"],
                    help="specify combiner type")

    ap.add_argument("--combiner_path", type=str, default=None,
                    help="used when combiner is dynamic_combiner")

    ap.add_argument("--top_k", type=int, default=None,
                    help="knn search size")
    
    ap.add_argument("--mixing_weight", type=float, default=None,
                    help="the weight of example-based distribution")

    ap.add_argument("--kernel", type=str, default=None, choices=["gaussian", "laplacian"],
                    help="used to compute similarity of query and retrieved examples based on distances")

    ap.add_argument("--bandwidth", type=float, default=None,
                    help="bandwidth in gaussian kernel or laplacian kernel")

    ap.add_argument("--index_path", type=str, default=None,
                    help="path of database index file")

    ap.add_argument("--token_map_path", type=str, default=None,
                    help="path of database token_map file")

    ap.add_argument("--embedding_path", type=str, default=None,
                    help="path of database embeddings file, used when mode == build_database or combiner == dynamic_combiner")

    ap.add_argument("--in_memory", type=str, default="True", choices=["True", "False"],
                    help="whether load embeddings file to memory, used when combiner == dynamic_combiner")

    ap.add_argument("--output_path", type=str,
                    help="path for saving translation output")

    ap.add_argument("--save_attention", action="store_true",
                    help="save attention visualizations")

    ap.add_argument("--division", type=str, default="train", choices=["train", "dev", "test"],
                    help="part of dataset, used when mode == build_database")

    args = ap.parse_args()

    combiner_cfg = {
        "type": args.combiner,
        "top_k": args.top_k,
        "mixing_weight": args.mixing_weight,
        "kernel": args.kernel,
        "bandwidth": args.bandwidth,
        "combiner_path": args.combiner_path,
        "index_path": args.index_path,
        "token_map_path": args.token_map_path,
        "embedding_path": args.embedding_path,
        "in_memory": args.in_memory == "True"
    }

    if args.mode == "train":
        train(cfg_file=args.config_path)
    elif args.mode == "combiner_train":
        combiner_train(cfg_file=args.config_path, ckpt=args.ckpt, combiner_cfg=combiner_cfg)
    elif args.mode == "test":
        test(cfg_file=args.config_path, ckpt=args.ckpt, combiner_cfg=combiner_cfg,
             output_path=args.output_path, save_attention=args.save_attention)
    elif args.mode == "build_database":
        from joeynmt.build_database import build_database
        build_database(cfg_file=args.config_path, ckpt=args.ckpt, division=args.division,
            index_path=args.index_path, embedding_path=args.embedding_path, token_map_path=args.token_map_path)
    elif args.mode == "analyze":
        analyze(cfg_file=args.config_path, ckpt=args.ckpt, combiner_cfg=combiner_cfg,
             output_path=args.output_path)
    elif args.mode == "score_translations":
        from joeynmt.prediction import score_translations
        score_translations(cfg_file=args.config_path, ckpt=args.ckpt, combiner_cfg=combiner_cfg,
             output_path=args.output_path)
    elif args.mode == "translate":
        translate(cfg_file=args.config_path, ckpt=args.ckpt,
                  output_path=args.output_path)
    else:
        raise ValueError("Unknown mode")


if __name__ == "__main__":
    main()
