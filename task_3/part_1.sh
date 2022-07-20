!#/usr/bin/env bash
sed -e 's/[^[:alpha:]]/ /g' dracula.txt | tr '\n' " " |  tr -s " " | tr " " '\n'| tr 'A-Z' 'a-z' | sort | uniq -c | sort -nr