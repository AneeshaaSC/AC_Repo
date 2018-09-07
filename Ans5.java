package assignment2;

import java.util.*;

public class Ans5 {

	public static void main(String args[])
	{
		Scanner scan=new Scanner(System.in);
		System.out.println("Enter temperature in Farenheit:");
		float t=scan.nextFloat();
		
		System.out.println("Temperature in Centigrade:"+(t-32)/1.8);
	}
}
