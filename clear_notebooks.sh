for k in $(find . -name "*.ipynb"); do echo $k;jupyter nbconvert --clear-output --inplace $k;  done
