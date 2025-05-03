#!/usr/bin/env sh

# Directory containing the directories of directories
validation_dataset="validation_data"

# Program to run
bin="target/release/sound"
model_save_file="long_training.ron"

total=0
num_correct=0

# Loop through all directories within the top directory
for dir in "$validation_dataset"/*; do
  if [ ! -d "$dir" ]; then
    echo "$dir není složka třídy!"
    exit 1
  fi

  class=$( basename "$dir" )

  echo "Running on class $class"
  $bin classify $model_save_file $dir/* > output.tmp

  # Use AWK to count the lines where the second word matches the search string
  match_count=$(awk -v search="$class" 'BEGIN{count = 0}{
    if ($2 == search) {
      count++
    }
  }
  END { print count }' output.tmp)

  lines_in_output=$(wc -l output.tmp | awk '{print $1}')
  total=$(echo "$total + $lines_in_output" | bc)
  num_correct=$(echo "$num_correct + $match_count" | bc)
done

rm output.tmp

accuracy=$(echo "scale=4; $num_correct / $total" | bc)

echo "Přesnost = $accuracy"
