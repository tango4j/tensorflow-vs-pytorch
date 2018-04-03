for file in *.ipynb; do
    echo "$file" "$(basename "$file" .ipynb).txt"
done
