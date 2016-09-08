#!/bin/bash

file="../bin/PIRWLS-train"
if [ -f "$file" ]
then
	echo "$file ."
else
	echo "$file not found. Please, build this code using the make command."
fi
