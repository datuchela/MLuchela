#!/bin/bash

VENV_DIR="./venv"
LAUNCH_SCRIPT="python ."

if [[ -f "${VENV_DIR}"/bin/activate ]]
then
	source "${VENV_DIR}"/bin/activate
else
	printf "\n%s\n" "${delimiter}"
	printf "\e[1m\e[31mERROR: Cannot activate python venv, aborting...\e[0m"
	printf "\n%s\n" "${delimiter}"
	exit 1
fi

$LAUNCH_SCRIPT
