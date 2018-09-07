package assignment8;

import java.util.*;

public class ans2 {

	public static void main(String args[])
	{
		Scanner sc=new Scanner(System.in);
		System.out.println("Enter you full name");
		String input=sc.nextLine();
		String name=" "+input;
		//System.out.println("Name:"+name);
		for (int i=0;i<name.length();i++)
		{
			if(name.charAt(i)==' ')
			{
				System.out.println(name.charAt(i+1));
			}
		}
		
		
	}
}
