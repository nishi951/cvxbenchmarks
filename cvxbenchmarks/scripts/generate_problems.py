from cvxbenchmarks.problem_generator import ProblemTemplate
from cvxbenchmarks.index import Index
import os

from cvxbenchmarks.scripts.utils import get_command_line_args

def main(args):
    template_files = [
        ("least_squares.j2", "least_squares_params.csv"),
        ("lasso.j2", "lasso_params.csv"),
        ("control.j2", "control_params.csv")
        # ("portfolio_opt.j2", "portfolio_opt_params.csv"),
        # ("lp.j2", "lp_params.csv"),
        # ("basis_pursuit.j2", "basis_pursuit_params.csv"),
        # ("chebyshev.j2", "chebyshev_params.csv"),
        # ("chebyshev_epigraph.j2", "chebyshev_params.csv"),
        # ("covsel.j2", "covsel_params.csv"),
        # ("fused_lasso.j2", "fused_lasso_params.csv"),
        # ("hinge_l1.j2", "hinge_l1_params.csv"),
        # ("hinge_l2.j2", "hinge_l2_params.csv"),
        # ("huber.j2", "huber_params.csv"),
        # ("infinite_push.j2", "infinite_push_params.csv"),
        # ("lasso.j2", "lasso_params.csv"),
        # ("least_abs_dev.j2", "least_abs_dev_params.csv"),
        # ("logreg_l1.j2", "logreg_l1_params.csv"),
        # ("max_gaussian.j2", "max_gaussian_params.csv"),
        # ("max_softmax.j2", "max_softmax_params.csv"),
        # ("oneclass_svm.j2", "oneclass_svm_params.csv"),
        # ("portfolio.j2", "portfolio_params.csv"),
        # ("qp.j2", "qp_params.csv"),
        # ("quantile.j2", "quantile_params.csv"),
        # ("robust_pca.j2", "robust_pca_params.csv"),
        # ("robust_svm.j2", "robust_svm_params.csv"),
        # ("tv_1d.j2", "tv_1d_params.csv")
    ]
    # print(args)
    templateDir = args.templateDir
    paramDir = args.paramDir
    problemDir = args.problemDir
    use_index = args.use_index
    print("Loading templates from: {}".format(templateDir))
    print("Loading parameters from: {}".format(paramDir))
    print("Target directory: {}".format(problemDir))

    # Collect problems from template files (1 per template)
    for templateName, paramName in template_files:
        paramFile = os.path.join(paramDir, paramName)
        templateFile = os.path.join(templateDir, templateName)
        templ = ProblemTemplate.from_file(templateFile, paramFile, templateName)
        templ.write_to_dir(problemDir)
        print("Wrote {} and {} to {}".format(templateName,
                                             paramFile,
                                             problemDir))

    # Generate index file
    index = Index(problemDir)
    index.write()
    index.write_latex(keys = ["num_scalar_variables",
                            "num_scalar_eq_constr",
                            "num_scalar_leq_constr"],
                      filename="index.tex")

if __name__ == '__main__':
    main(get_command_line_args())
