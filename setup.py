from distutils.core import setup



setup(
    name='pbtspot',
    version='0.1.0',
    author='Sourav Chatterjee',
    author_email='souravc83@gmail.com',
    packages=['pbtspot','pbtspot.adaboost','pbtspot.adaboost.features',\
              'pbtspot.adaboost.generateimage'],
    package_dir={'pbtspot':'src/pbtspot',
                  'pbtspot.adaboost':'src/pbtspot/adaboost',
                  'pbtspot.adaboost.features':'src/pbtspot/adaboost/features',
                  'pbtspot.adaboost.generateimage':'src/pbtspot/adaboost/generateimage'
                  },
    description='Probabilistic Boosting Tree for Spot Detection: \
                 A python library that uses Probabilistic Boosting Tree \
                 and Viola Jones Agorithm to detect spots in images',
    requires=[ "python(==2.7)",
               "matplotlib(>=1.1)",
                "numpy",
                "json"
                ],
    provides=["pbtspot"]    
)
