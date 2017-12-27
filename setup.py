from setuptools import setup, find_packages

setup(
    name = 'cvxbenchmarks',
    version = '0.0.1',
    description = 'A testing framework for convex solvers.',
    author = 'Mark Nishimura, Steven Diamond, Stephen Boyd',
    author_email = 'markn1@stanford.edu',
    packages = ['cvxbenchmarks',
                'cvxbenchmarks.cvx',
                'cvxbenchmarks.tests',
                'cvxbenchmarks.scripts'],
    package_dir = {'cvxbenchmarks': 'cvxbenchmarks'},
    zip_safe = False,
    use_2to3 = True,
    install_requires = ['cvxpy >= 0.4.8',
                        'pandas >= 0.18.1',
                        'tabulate >= 0.7.5',
                        'Jinja2 >= 2.9.6',
                        'matplotlib >= 2.0.2',
                        'multiprocess >= 0.70.5',
                        'ruamel.yaml < 0.15',
                        # Unit Testing
                        'pytest >= 3.3.1',
                        'pytest-cov >= 2.5.1',
                        'mock >= 2.0.0'
                        ],
    entry_points = {
        'console_scripts': ['cvxbench=cvxbenchmarks.command_line:main']
    }
)
