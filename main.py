"""Command-line entry point for the season-split F1 pit-stop pipeline."""

import argparse

from src.pipeline import run_evaluate, run_preprocess, run_train, run_tune


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='F1 Pit-Stop HMM Pipeline')
    parser.add_argument('--preprocess', action='store_true', help='Build features from Kaggle CSVs')
    parser.add_argument(
        '--tune',
        action='store_true',
        help='Tune thresholds and HMM states on 2019 only (fit candidates on 2011-2018)',
    )
    parser.add_argument('--train', action='store_true', help='Train on seasons before --test_year')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate the held-out season')
    parser.add_argument(
        '--test_year',
        type=int,
        default=2020,
        help='Held-out season (default: 2020)',
    )
    parser.add_argument(
        '--compounds',
        nargs='+',
        default=['SOFT', 'MEDIUM', 'HARD'],
        help='Tyre compounds to train',
    )
    parser.add_argument('--n_iter', type=int, default=50, help='Baum-Welch iterations')
    parser.add_argument(
        '--n_states',
        type=int,
        choices=[2, 3, 4],
        default=3,
        help='HMM state count for non-final training (final training uses 2019 tuning)',
    )
    parser.add_argument('--tolerance', type=int, default=2, help='Pit matching tolerance in laps')
    parser.add_argument('--drivers', nargs='+', help='Driver codes to evaluate')
    parser.add_argument('--plot_drivers', nargs='+', help='Driver codes to plot')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not (args.preprocess or args.tune or args.train or args.evaluate):
        raise SystemExit('Choose --preprocess, --tune, --train, or --evaluate.')
    if args.preprocess:
        run_preprocess()
    if args.tune:
        run_tune(n_iter=args.n_iter, compounds=args.compounds)
    if args.train:
        run_train(
            test_year=args.test_year,
            compounds=args.compounds,
            n_iter=args.n_iter,
            n_states=args.n_states,
        )
    if args.evaluate:
        run_evaluate(
            test_year=args.test_year,
            tolerance=args.tolerance,
            drivers=args.drivers,
            plot_drivers=args.plot_drivers,
        )


if __name__ == '__main__':
    main()
