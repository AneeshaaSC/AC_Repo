package assignment4;

import java.util.*;

class Company {
	String companyname;
	String address;
	int no_of_employees;
	int no_of_departments;
	void gen_details(String n,String a, int noe, int nod)
	{
		companyname=n;
		address=a;
		no_of_employees=noe;
		no_of_departments=nod;
		System.out.println("Company name: "+companyname+"\nAddress: "+address+"\nNumber of employees: "+no_of_employees+"\nNumber of departments: "+no_of_departments);
		}
	}

public class ans1 {

	public static void main(String args[]) {
		Company comp=new Company();
		
		Scanner sc=new Scanner(System.in);
		

		
		System.out.println("Enter Company Name");
		String name=sc.next();
		
		System.out.println("Enter number of employees");
		int noe=sc.nextInt();
		
		System.out.println("Enter number of departments");
		int nod=sc.nextInt();
		
		System.out.println("Enter Company Address");
		String addr=sc.next();
		

		

		
		comp.gen_details(name,addr,noe,nod);
	}
	
	
}
