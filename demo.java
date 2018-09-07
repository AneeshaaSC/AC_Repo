package assignment2;

import java.util.Scanner;

public class demo {
	
public static void main(String args[])
{
	Scanner scan = new Scanner(System.in);
	System.out.println("Enter Distance travelled in Kilometers: ");
	float distance=scan.nextFloat();
	
	System.out.println("Enter Time taken in hours: ");
	float timehr=scan.nextFloat();
	
	float speed;
	
	speed=distance/timehr;
	
	System.out.println("Speed of the Vehicle is: "+speed+" km/hr");
	
}
}
