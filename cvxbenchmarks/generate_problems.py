import ProblemGenerator as pg
import os

template_files = [
    ("least_squares.j2", "least_squares_params.csv"),
    ("lasso.j2", "lasso_params.csv"),
    ("control.j2", "control_params.csv"),
    ("portfolio_opt.j2", "portfolio_opt_params.csv"),
    ("lp.j2", "lp_params.csv"),
    ("basis_pursuit.j2", "basis_pursuit_params.csv"),
    ("chebyshev.j2", "chebyshev_params.csv"),
    ("covsel.j2", "covsel_params.csv"),
    # ("fused_lasso.j2", "fused_lasso_params.csv"),
    # ("hinge_l1.j2", "hinge_l1_params.csv"),
    # ("hinge_l2.j2", "hinge_l2_params.csv"),
    # ("huber.j2", "huber_params.csv"),
    # ("infinite_push.j2", "infinite_push_params.csv"),
    # ("lasso_ep.j2", "lasso_ep_params.csv"),
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

templateDir = os.path.join("lib", "data", "cvxbenchmarks")

# Generate problems from templates
for templateFile, paramFile in template_files:
    paramFile = os.path.join(templateDir, paramFile)
    templ = pg.ProblemTemplate(templateFile, paramFile, templateDir)
    templ.write_to_dir("problems")

# Generate index file
index = pg.Index("problems")
index.write()

