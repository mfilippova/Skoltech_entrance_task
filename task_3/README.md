# Task 3

Команды к заданиям находятся в файлах `part_1.sh` и `part_2.sh` соответственно.

Запустить через `docker exec ...` у меня не получилось (возможно, получится у вас :) ). Но команды верно отрабатывают, если их запускать в командной строке внутри контейнера.

Пример запуска контейнера и команд:

```
docker build . -f Dokerfile -t task_3
docker run --rm -it task_3 /bin/bash
sed -e 's/[^[:alpha:]]/ /g' dracula.txt | tr '\n' " " |  tr -s " " | tr " " '\n'| tr 'A-Z' 'a-z' | sort | uniq -c | sort -nr
sed -e 's/[^[:alpha:]]/ /g' dracula.txt | tr '\n' " " |  tr -s " " | tr " " '\n'| tr 'A-Z' 'a-z' | sort | uniq -c | sort -nr | head -n 10 | awk '{print $2}' |  xargs touch
```