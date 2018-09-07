package assignment8;
import java.util.*;

public class ans1 {
	public static void main(String args[])
	{
		Scanner sc=new Scanner(System.in);
		System.out.println("Enter any String Aneeshaa");
		String a=sc.next();
		sc.nextLine();
		System.out.println("Enter any String you'd like to replace");
		String b=sc.nextLine();
		//sc.nextLine();
		System.out.println("What would you like to replace it with?");
		String c=sc.next();
		
		String d=a.replace(b, c);
		System.out.println("Your new string is:"+d);
	}

}
