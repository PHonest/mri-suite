for f in *.tar.gz; do
  name="${f%.tar.gz}"              # strip .tar.gz extension
  mkdir -p "$name"_tmp
  tar -xzf "$f" -C "$name"_tmp     # note the "z" for gzip
  (cd "$name"_tmp && zip -r "../$name.zip" .)
  rm -rf "$name"_tmp
done