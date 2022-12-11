if [ -d data/pascalvoc ];
then 
    echo "Data/pascalvoc already exists... Download?"
else
    mkdir data/pascalvoc
fi
    
cd src && python data.py