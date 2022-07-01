test -e run_db.txt && mv run_db.txt run_db.bak

for dir in UNet_2022*
do
   #whatever you need with "$dir"
   { echo $dir": " ; tr -d '\n' < $dir"/command_line_args.txt" ; echo ''; } >> run_db.txt
   #echo -e '\n' >> run_db.txt
done
