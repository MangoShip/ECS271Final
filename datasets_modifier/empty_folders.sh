#!/bin/bash
directory="../balanced_datasets"
classes=("billete" "knife" "monedero" "pistol" "smartphone" "tarjeta")

for class in ${classes[@]}
do
    rm $directory/$class/*
done