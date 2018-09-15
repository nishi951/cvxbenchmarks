# command_line.py
# Entry point for command line cvxbench interface

from cvxbenchmarks.scripts.utils import get_command_line_args

def main():
    """Hook for command line interface."""

    args = get_command_line_args()

    if args.script == "analyze":
        from cvxbenchmarks.scripts import analyze_results
        analyze_results.main(args)

    elif args.script == "generate":
        from cvxbenchmarks.scripts import generate_problems
        generate_problems.main(args)

    elif args.script == "run":
        from cvxbenchmarks.scripts import run_tests
        run_tests.main(args)

    elif args.script == "render":
        from cvxbenchmarks.scripts import render_template
        render_template.main(args)

    elif args.script == "view":
        from cvxbenchmarks.scripts import view
        view.main(args)

if __name__ == '__main__':
    main()