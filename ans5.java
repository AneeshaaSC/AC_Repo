package assignment7;

import java.util.*;

class ownException extends Exception
{
	ownException(String msg)
	{
		super(msg);
	}
}


public class ans5 {
	
	public void checkarg(int var)
	{
		if (var==0)
		{
			throw new IllegalArgumentException();
		}
	}
	

	public static void main(String args[]) throws IllegalArgumentException,ArithmeticException,ownException
	{
		ans5 ans5obj=new ans5();
		Scanner sc=new Scanner(System.in);
		
		System.out.println("Enter an integer: ");
		int a=sc.nextInt();
		try 
		{
			ans5obj.checkarg(a);
		}
		catch(IllegalArgumentException e)
		{
			System.out.println("This integer cannot be zero");
		}

		float b=10/a;
		System.out.println("10 divided by "+a+" is: "+b);
		
		System.out.println("Enter an integer: ");
		int c=sc.nextInt();
		if (c>0)
		{
			throw new ownException("Integer you entered is greater than 0");
		}
		
		
	}

}
