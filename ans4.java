package assignment7;

import java.util.*;
//import java.lang.*;

public class ans4 {
	public static void main(String args[])
	{
		int c;
		do {
		Scanner sc=new Scanner(System.in);
		System.out.println("Enter first integer:");
		int a=sc.nextInt();
		System.out.println("Enter second integer:");
		int b=sc.nextInt();	
		c=a+b;
		System.out.println("Sum of the integers:"+c);
		if (c>10) 
		{
			throw new ArithmeticException("Sum goes beyond 10"); 
		}
		}while(c<=10);


	}

}
