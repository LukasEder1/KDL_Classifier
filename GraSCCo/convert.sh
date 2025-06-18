for file in *.txt; do
  libreoffice --headless --convert-to pdf "$file"
done
