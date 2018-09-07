package assignment7;

import java.util.*;

class weightexception extends Exception
{
	weightexception(String msg)
	{
		super(msg);
	}
	
}

public class ans6 {

	public static void main(String args[])
	{
		Scanner sc = new Scanner(System.in);
		System.out.println("Enter product weight in grams");
		Float w=sc.nextFloat();
		try
		{
		if (w<1000)
		{
			throw new weightexception("Weight is less than 1000gms");
			
		}
		else 
		{
			System.out.println("Weight of the product is: "+w);
		}
		}
		catch(weightexception e)
		{
			System.out.println("Argument is not acceptable");
			System.out.println(e.getMessage());
		}
	}
}
