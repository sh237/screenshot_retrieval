query_path=$(readlink -f $(dirname $0)/../config/query.json)

cmd="curl -X POST \
    -F query=@$query_path \
    http://localhost:8080/oneshot"

exec time -p $cmd
