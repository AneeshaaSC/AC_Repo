package assignment4;

class Account
{
	String a_name;
	String a_type;
	float ini_amt;
	int a_num;
	String a_addr;
	
	Account(String name,int a_num,float amt)
	{
		a_name=name;
		this.a_num=a_num;
		ini_amt=amt;
	}
	
	Account(String name,int a_num,String addr,String a_type)
	{
		a_name=name;
		this.a_num=a_num;
		a_addr=addr;
		this.a_type=a_type;
	}
	void printing()
	{
		System.out.println("Account Holder name: "+a_name);
		System.out.println("Account Number: "+a_num);
		
		if (a_addr!=null) {
		System.out.println("Account Holder Address: "+a_addr);
		}
		
		if (ini_amt!=0.0f) {
		System.out.println("Initial Amount: "+ini_amt);
		}
		
		if (a_type!=null) {
		System.out.println("Account Type: "+a_type);
		}
		
	}

}

public class ans4 {
	public static void main(String args[])
	{
		Account a1=new Account("Akshay",123,50000.00f);
		Account a2=new Account("Aneeshaa",456,"V'pura","Current");
		a1.printing();
		a2.printing();
		
		
	}

}
