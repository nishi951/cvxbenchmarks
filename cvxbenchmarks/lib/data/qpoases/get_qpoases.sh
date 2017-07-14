# Download and unzip qpoases
# Mark Nishimura 2017

if ! [ -s chain80.tar.gz ]; then
    wget http://www.qpoases.org/onlineQP/downloads/chain80.tar.gz
    tar -xvf chain80.tar.gz
fi
if ! [ -s chain80w.tar.gz ]; then
    wget http://www.qpoases.org/onlineQP/downloads/chain80w.tar.gz
    tar -xvf chain80w.tar.gz
fi
if ! [ -s diesel.tar.gz ]; then
    wget http://www.qpoases.org/onlineQP/downloads/diesel.tar.gz
    tar -xvf diesel.tar.gz
fi
if ! [ -s crane.tar.gz ]; then
    wget http://www.qpoases.org/onlineQP/downloads/crane.tar.gz
    tar -xvf crane.tar.gz
fi
if ! [ -s CDU.tar.gz ]; then
    wget http://www.qpoases.org/onlineQP/downloads/CDU.tar.gz
    tar -xvf CDU.tar.gz
fi



