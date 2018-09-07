package assignment3;

import java.util.*;

public class ans1 {

	public static void main (String args[])
	{
		Scanner scan=new Scanner(System.in);
		System.out.println("Enter the two numbers:");
		int num1=scan.nextInt();
		int num2=scan.nextInt();
		
		int largest=((num1 > num2) ? num1 : num2);
		System.out.println("Largest number is: "+largest);
	}
	
}
