package assignment2;

import java.util.Scanner;

public class Ans2 {

	public static void main(String args[])
	{
		Scanner scan = new Scanner(System.in);
		System.out.println("Enter time in hours:");
		int timehr=scan.nextInt();
		
		int timemin=timehr*60;
		System.out.println("Time in minutes:"+timemin);
		
		int timesec=timemin*60;
		System.out.println("Time in seconds:"+timesec);
		
		
	}
}
