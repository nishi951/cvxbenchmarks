from setuptools import setup, find_packages

setup(
    name = 'CVXBenchmarks',
    version = '0.0.1',
    description = 'A testing framework for convex solvers.',
    author = 'Mark Nishimura, Steven Diamond, Stephen Boyd',
    author_email = 'markn1@stanford.edu',
    packages = ['cvxbenchmarks',
                'cvxbenchmarks.tests',
                'cvxbenchmarks.scripts'],
    package_dir = {'cvxbenchmarks': 'cvxbenchmarks'},
    zip_safe = False,
    use_2to3 = True,
    install_requires = ['cvxpy >= 0.4.2',
                        'pandas >= 0.18.1',
                        'tabulate >= 0.7.5',
                        ],
    entry_points = {
        'console_scripts': ['cvxbench=cvxbenchmarks.command_line:main']
    }
)
