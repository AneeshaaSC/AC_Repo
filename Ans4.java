package assignment2;

import java.util.Scanner;

public class Ans4 {
	public static void main(String args[])
	{
		Scanner scan=new Scanner(System.in);
		System.out.println("Enter Current in Amps:");
		float I=scan.nextFloat();

		System.out.println("Enter Resistance in Ohms:");
		float R=scan.nextFloat();
		
		System.out.println("Power of the bulb in Watts:"+Math.pow(I, 2)*R);
		
		
		
	}

}
