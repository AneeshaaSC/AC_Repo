package assignment3;

import java.util.*;

public class ans2 {

	public static void main (String args[])
	{
		Scanner scan=new Scanner(System.in);
		System.out.println("Enter a number of your choice:");
		int num=scan.nextInt();
		
		num=num+8;
		System.out.println("adding 8: "+num);
		
		if (num % 5 == 0){
			System.out.println("It is divisible by 5");
		}
		else {
			System.out.println("It is NOT divisible by 5");
		}

		if (num % 7 == 0){
			System.out.println("It is divisible by 7");
		}
		else {
			System.out.println("It is NOT divisible by 7");
		}
		
		if (num%7==0 && num%5==0) {
			System.out.println("It is divisible by both 5 and 7");
		}
	}
}
