package assignment2;

import java.util.Scanner;

public class Ans3 {

	public static void main(String args[])
	{
		Scanner scan=new Scanner(System.in);
		System.out.println("Enter Principal Amount:");
		float p=scan.nextFloat();

		System.out.println("Enter time in years:");
		float t=scan.nextFloat();
		
		System.out.println("Enter rate of interest:");
		float r=scan.nextFloat();
		
		float S_I=(p*t*r)/100;
		System.out.println("Simple Interest Calculated is:"+S_I);
		
	}
	
}
