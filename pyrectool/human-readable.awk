#!/usr/bin/awk -f

{	sum = $1
	hum[1000**3]="g";hum[1000**2]="m";hum[1000]="k";hum[0]="";
	for (x=1000**3; x>=0; x/=1000){ 
        	if (sum>=x) { printf "%i%s",sum/x,hum[x];break }
}}
