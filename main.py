"""Command-line entry point for the season-split F1 pit-stop pipeline."""

import argparse

from src.pipeline import run_evaluate, run_preprocess, run_train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='F1 Pit-Stop HMM Pipeline')
    parser.add_argument('--preprocess', action='store_true', help='Fetch laps and build features')
    parser.add_argument('--train', action='store_true', help='Train on seasons before --test_year')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate the held-out season')
    parser.add_argument('--year', type=int, help='Single season for preprocessing')
    parser.add_argument(
        '--years',
        nargs='+',
        type=int,
        help='Complete seasons to preprocess (e.g. --years 2020 2021 2022 2023)',
    )
    parser.add_argument('--gp', nargs='+', help='Grand Prix names for preprocessing')
    parser.add_argument(
        '--test_year',
        type=int,
        help='Held-out season (defaults to latest season in the processed CSV)',
    )
    parser.add_argument(
        '--compounds',
        nargs='+',
        default=['SOFT', 'MEDIUM', 'HARD'],
        help='Tyre compounds to train',
    )
    parser.add_argument('--n_iter', type=int, default=50, help='Baum-Welch iterations')
    parser.add_argument('--n_states', type=int, default=3, help='Must remain 3 for tyre states')
    parser.add_argument('--tolerance', type=int, default=2, help='Pit matching tolerance in laps')
    parser.add_argument('--drivers', nargs='+', help='Driver codes to evaluate')
    parser.add_argument('--plot_drivers', nargs='+', help='Driver codes to plot')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_all = not (args.preprocess or args.train or args.evaluate)

    if run_all or args.preprocess:
        run_preprocess(year=args.year, gp=args.gp, years=args.years)
    if run_all or args.train:
        run_train(
            test_year=args.test_year,
            compounds=args.compounds,
            n_iter=args.n_iter,
            n_states=args.n_states,
        )
    if run_all or args.evaluate:
        run_evaluate(
            test_year=args.test_year,
            tolerance=args.tolerance,
            drivers=args.drivers,
            plot_drivers=args.plot_drivers,
        )


if __name__ == '__main__':
    main()
