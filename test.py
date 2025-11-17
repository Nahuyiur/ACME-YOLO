import warnings

from val import build_parser, run_validation

warnings.filterwarnings("ignore")


def main() -> None:
    parser = build_parser(default_split="test", default_project="runs/test")
    args = parser.parse_args()
    run_validation(args)


if __name__ == "__main__":
    main()

