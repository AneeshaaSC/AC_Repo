package assignment7;

import java.util.*;

public class ans1 {
	public static void main(String args[])
	{
	Scanner sc=new Scanner(System.in);
	System.out.println("Enter any number Aneeshaa");
	float a = sc.nextFloat();
	float c;
	
	for(int i=5;i>=0;i--)
	{
		try 
		{
		c=a/i;
		if()
		System.out.println(a+ " divided by "+i+" is "+c);
		}
		catch(ArithmeticException ae)
		{
			System.out.println("How can you diviide by zero?");
		}
		
	}

	
}
}
